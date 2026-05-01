from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
import triton
from triton.runtime.errors import OutOfResources

_TTGIR_VALUE_RE = r"%[A-Za-z0-9_.$]+(?:#\d+)?"


def _format_tune_args(tune_args: Any | None) -> str | None:
    if tune_args is None:
        return None
    if is_dataclass(tune_args):
        values = asdict(tune_args)
    elif isinstance(tune_args, Mapping):
        values = dict(tune_args)
    else:
        values = {
            key: getattr(tune_args, key)
            for key in dir(tune_args)
            if key.isupper() and not key.startswith("_")
        }
    formatted = ", ".join(f"{key}={value}" for key, value in values.items() if value is not None)
    return formatted or None


def _format_input_shapes(input_shapes: Any | None) -> str | None:
    if input_shapes is None:
        return None
    if isinstance(input_shapes, Mapping):
        values = dict(input_shapes)
    else:
        values = {"input_shapes": input_shapes}
    formatted = ", ".join(f"{key}={value}" for key, value in values.items() if value is not None)
    return formatted or None

def capture_compiled_artifact_paths(compiled: object, kernel_name: str) -> dict[str, str]:
    asm = getattr(compiled, "asm", None)
    if asm is None:
        return {}

    artifact_dir = Path(tempfile.gettempdir()) / "auto_flash_atten_triton_artifacts"
    artifact_dir /= f"{kernel_name}_{getattr(compiled, 'hash', 'unknown')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    path_map: dict[str, str] = {}
    for ext in ("ptx", "ttir", "ttgir"):
        artifact = asm.get(ext)
        if artifact is None:
            continue
        artifact_path = artifact_dir / f"{kernel_name}.{ext}"
        artifact_path.write_text(str(artifact), encoding="utf-8")
        path_map[f"compiled_{ext}_path"] = str(artifact_path)
    return path_map


def _lookup_named_value(container: Any | None, key: str) -> Any | None:
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def _estimate_loop_iterations(input_shapes: Any | None, tune_args: Any | None) -> int:
    block_n = _lookup_named_value(tune_args, "BLOCK_N")
    k_shape = _lookup_named_value(input_shapes, "k")
    if not isinstance(block_n, int) or block_n <= 0:
        return 1
    if not isinstance(k_shape, tuple) or len(k_shape) < 3:
        return 1
    seq_len = k_shape[2]
    if not isinstance(seq_len, int) or seq_len <= 0:
        return 1
    return max(1, triton.cdiv(seq_len, block_n))


def _load_ttgir_context(ptx_path: str | None) -> dict[str, object]:
    if not ptx_path:
        return {}

    ptx_file = Path(ptx_path)
    if not ptx_file.exists():
        return {}

    ttgir_path = ptx_file.with_suffix(".ttgir")
    ttir_path = ptx_file.with_suffix(".ttir")
    if not ttgir_path.exists():
        return {}

    ttgir_text = ttgir_path.read_text(encoding="utf-8")
    defs: dict[str, str] = {}
    defs_in_loop: set[str] = set()
    def_order: dict[str, int] = {}
    use_map: dict[str, list[str]] = {}
    alias_map: dict[str, list[str]] = {}
    iter_arg_initial_values: dict[str, str] = {}
    loop_switches: list[dict[str, str]] = []
    loc_defs: dict[str, str] = {}
    source_cache: dict[str, list[str]] = {}
    dot_ops: list[tuple[str, str, str]] = []
    load_variables: list[tuple[str, str | None]] = []
    active_loop_iter_args: list[str] | None = None
    active_loop_results: list[str] | None = None
    active_loop_iter_arg_pairs: list[tuple[str, str]] | None = None

    for line in ttgir_text.splitlines():
        stripped = line.strip()
        loc_match = re.match(r"^(#loc\d+)\s*=\s*(.+)$", stripped)
        if loc_match:
            loc_defs[loc_match.group(1)] = loc_match.group(2)
            continue
        match = re.match(r"^(%\S+)\s*=\s*(.+)$", stripped)
        if not match:
            if " = scf.for " in stripped and "iter_args(" in stripped:
                iter_args_match = re.search(r"iter_args\((.+?)\)\s*->", stripped)
                active_loop_iter_args = []
                active_loop_results = []
                active_loop_iter_arg_pairs = []
                loop_head_match = re.match(r"^(%[A-Za-z0-9_.$]+):(\d+)\s*=\s*scf\.for", stripped)
                if loop_head_match:
                    loop_base = loop_head_match.group(1)
                    loop_count = int(loop_head_match.group(2))
                    active_loop_results = [f"{loop_base}#{idx}" for idx in range(loop_count)]
                if iter_args_match:
                    for iter_arg, initial_value in re.findall(
                        rf"({_TTGIR_VALUE_RE})\s*=\s*({_TTGIR_VALUE_RE})",
                        iter_args_match.group(1),
                    ):
                        active_loop_iter_args.append(iter_arg)
                        active_loop_iter_arg_pairs.append((iter_arg, initial_value))
                        iter_arg_initial_values[iter_arg] = initial_value
                        alias_map.setdefault(iter_arg, []).append(initial_value)
                        use_map.setdefault(initial_value, []).append(iter_arg)
                continue
            if stripped.startswith("scf.yield ") and active_loop_iter_args:
                yielded_prefix = stripped.split(":", 1)[0]
                yielded_values = re.findall(_TTGIR_VALUE_RE, yielded_prefix)
                for yielded_value, iter_arg in zip(yielded_values, active_loop_iter_args):
                    alias_map.setdefault(iter_arg, []).append(yielded_value)
                    use_map.setdefault(yielded_value, []).append(iter_arg)
                    initial_value = iter_arg_initial_values.get(iter_arg, "")
                    if initial_value and yielded_value != initial_value:
                        loop_switches.append(
                            {
                                "iter_arg": iter_arg,
                                "source_value": yielded_value,
                                "target_value": initial_value,
                            }
                        )
                if active_loop_results:
                    for yielded_value, loop_result in zip(yielded_values, active_loop_results):
                        alias_map.setdefault(loop_result, []).append(yielded_value)
                continue
            if stripped == "}" or stripped.startswith("} loc("):
                active_loop_iter_args = None
                active_loop_results = None
                active_loop_iter_arg_pairs = None
            continue
        var_name = match.group(1)
        rhs = match.group(2)
        def_order[var_name] = len(def_order)
        if rhs.startswith("scf.for ") and "iter_args(" in rhs:
            iter_args_match = re.search(r"iter_args\((.+?)\)\s*->", rhs)
            active_loop_iter_args = []
            active_loop_results = []
            active_loop_iter_arg_pairs = []
            loop_head_match = re.match(r"^(%[A-Za-z0-9_.$]+):(\d+)$", var_name)
            if loop_head_match:
                loop_base = loop_head_match.group(1)
                loop_count = int(loop_head_match.group(2))
                active_loop_results = [f"{loop_base}#{idx}" for idx in range(loop_count)]
            if iter_args_match:
                for iter_arg, initial_value in re.findall(
                    rf"({_TTGIR_VALUE_RE})\s*=\s*({_TTGIR_VALUE_RE})",
                    iter_args_match.group(1),
                ):
                    active_loop_iter_args.append(iter_arg)
                    active_loop_iter_arg_pairs.append((iter_arg, initial_value))
                    iter_arg_initial_values[iter_arg] = initial_value
                    alias_map.setdefault(iter_arg, []).append(initial_value)
                    use_map.setdefault(initial_value, []).append(iter_arg)
        defs[var_name] = rhs
        if active_loop_iter_args is not None and not rhs.startswith("scf.for "):
            defs_in_loop.add(var_name)
        for dependency in re.findall(_TTGIR_VALUE_RE, rhs):
            use_map.setdefault(dependency, []).append(var_name)
        if rhs.startswith("tt.load "):
            load_loc_match = re.search(r"loc\((#loc\d+)\)", rhs)
            load_variables.append((var_name.lstrip("%"), load_loc_match.group(1) if load_loc_match else None))
        if rhs.startswith("tt.dot "):
            operands = re.findall(_TTGIR_VALUE_RE, rhs.split(":", 1)[0])
            if len(operands) >= 3:
                dot_ops.append((var_name, operands[0], operands[1]))

    def resolve_loc_name(loc_id: str | None, seen: set[str] | None = None) -> str | None:
        if loc_id is None:
            return None
        if seen is None:
            seen = set()
        if loc_id in seen:
            return None
        seen.add(loc_id)
        expr = loc_defs.get(loc_id)
        if expr is None:
            return None

        quoted_name = re.match(r'loc\("([^"]+)"\(#loc\d+\)\)', expr)
        if quoted_name:
            return quoted_name.group(1)

        file_loc = re.match(r'loc\("([^"]+)":(\d+):(\d+)\)', expr)
        if file_loc:
            source_path = file_loc.group(1)
            line_no = int(file_loc.group(2))
            if source_path not in source_cache:
                try:
                    source_cache[source_path] = Path(source_path).read_text(encoding="utf-8").splitlines()
                except OSError:
                    source_cache[source_path] = []
            source_lines = source_cache[source_path]
            if 1 <= line_no <= len(source_lines):
                source_line = source_lines[line_no - 1].strip()
                assign_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*tl\.load\b", source_line)
                if assign_match:
                    return assign_match.group(1)
            return None

        for nested_loc_id in re.findall(r"#loc\d+", expr):
            resolved = resolve_loc_name(nested_loc_id, seen)
            if resolved is not None:
                return resolved
        return None

    return {
        "ptx_path": str(ptx_file),
        "ttgir_path": str(ttgir_path),
        "ttir_path": (str(ttir_path) if ttir_path.exists() else None),
        "ttgir_text": ttgir_text,
        "defs": defs,
        "defs_in_loop": defs_in_loop,
        "def_order": def_order,
        "use_map": use_map,
        "alias_map": alias_map,
        "iter_arg_initial_values": iter_arg_initial_values,
        "loop_switches": loop_switches,
        "loc_defs": loc_defs,
        "load_variables": load_variables,
        "dot_ops": dot_ops,
        "resolve_loc_name": resolve_loc_name,
    }


def find_all_variable_transfer(
    ptx_path: str | None,
    *,
    input_shapes: Any | None = None,
    tune_args: Any | None = None,
) -> list[dict[str, str]]:
    context = _load_ttgir_context(ptx_path)
    if not context:
        return []

    defs = dict(context["defs"])
    defs_in_loop = set(context["defs_in_loop"])
    def_order = dict(context["def_order"])
    alias_map = dict(context["alias_map"])
    iter_arg_initial_values = dict(context["iter_arg_initial_values"])
    load_variables = list(context["load_variables"])
    resolve_loc_name = context["resolve_loc_name"]
    load_name_map = {
        f"%{raw_variable}": (resolve_loc_name(loc_id) or raw_variable)  # type: ignore[operator]
        for raw_variable, loc_id in load_variables
    }
    iter_arg_slot_names = {
        iter_arg: load_name_map.get(initial_value, initial_value.lstrip("%"))
        for iter_arg, initial_value in iter_arg_initial_values.items()
    }
    loop_iteration_count = _estimate_loop_iterations(input_shapes, tune_args)
    loop_orders = [order for var_name, order in def_order.items() if var_name in defs_in_loop]
    loop_start_order = min(loop_orders) if loop_orders else None
    loop_end_order = max(loop_orders) if loop_orders else None
    loop_span = ((loop_end_order - loop_start_order + 1) if loop_start_order is not None and loop_end_order is not None else 0)

    def resolve_affiliated_names(
        source: str,
        seen: set[str] | None = None,
        prefer_current_slot_name: bool = False,
    ) -> set[str]:
        if prefer_current_slot_name and source in iter_arg_slot_names:
            return {iter_arg_slot_names[source]}
        if source in load_name_map:
            return {load_name_map[source]}
        if seen is None:
            seen = set()
        if source in seen:
            return set()
        seen.add(source)
        resolved_names: set[str] = set()
        for alias_source in alias_map.get(source, []):
            resolved_names.update(
                resolve_affiliated_names(
                    alias_source,
                    seen,
                    prefer_current_slot_name=prefer_current_slot_name,
                )
            )
        rhs = defs.get(source, "")
        for dependency in re.findall(_TTGIR_VALUE_RE, rhs):
            resolved_names.update(
                resolve_affiliated_names(
                    dependency,
                    seen,
                    prefer_current_slot_name=prefer_current_slot_name,
                )
            )
        return resolved_names

    transfer_entries: list[dict[str, str]] = []

    def add_transfer(
        variable_name: str | None,
        transfer_type: str,
        role: str,
        detail: str,
        occurrence_id: str,
        execution_scope: str,
    ) -> None:
        if not variable_name:
            return
        base_order = def_order.get(occurrence_id, -1)
        if execution_scope == "loop" and loop_start_order is not None and loop_span > 0:
            for iteration_idx in range(loop_iteration_count):
                transfer_entries.append(
                    {
                        "variable_name": variable_name,
                        "transfer_type": transfer_type,
                        "role": role,
                        "detail": detail,
                        "occurrence_id": occurrence_id,
                        "execution_scope": execution_scope,
                        "iteration_index": str(iteration_idx),
                        "instruction_order": str(
                            loop_start_order + iteration_idx * loop_span + (base_order - loop_start_order)
                        ),
                    }
                )
            return
        adjusted_order = base_order
        if (
            execution_scope == "function"
            and loop_start_order is not None
            and loop_end_order is not None
            and base_order > loop_end_order
            and loop_span > 0
        ):
            adjusted_order = base_order + (loop_iteration_count - 1) * loop_span
        transfer_entries.append(
            {
                "variable_name": variable_name,
                "transfer_type": transfer_type,
                "role": role,
                "detail": detail,
                "occurrence_id": occurrence_id,
                "execution_scope": execution_scope,
                "iteration_index": "0",
                "instruction_order": str(adjusted_order),
            }
        )

    for raw_variable, _loc_id in load_variables:
        load_var = f"%{raw_variable}"
        add_transfer(
            load_name_map[load_var],
            "global_to_register",
            "target",
            "tt.load produced this value from global memory into a register-backed SSA tensor",
            load_var,
            ("loop" if load_var in defs_in_loop else "function"),
        )

    for var_name, rhs in defs.items():
        dependencies = re.findall(_TTGIR_VALUE_RE, rhs)
        if "async_copy_global_to_local" in rhs:
            for local_name in resolve_affiliated_names(var_name, prefer_current_slot_name=True):
                add_transfer(
                    local_name,
                    "global_to_shared_memory",
                    "target",
                    "async global-to-local copy placed this value directly into shared memory",
                    var_name,
                    ("loop" if var_name in defs_in_loop else "function"),
                )
            continue
        if "local_alloc" in rhs and dependencies:
            for source_name in resolve_affiliated_names(dependencies[0], prefer_current_slot_name=True):
                add_transfer(
                    source_name,
                    "register_to_shared_memory",
                    "source",
                    "local_alloc staged this register-backed value into shared memory",
                    var_name,
                    ("loop" if var_name in defs_in_loop else "function"),
                )
            continue
        if "local_load" in rhs and dependencies:
            target_names = resolve_affiliated_names(
                dependencies[0],
                prefer_current_slot_name=True,
            ) or resolve_affiliated_names(var_name, prefer_current_slot_name=True)
            for target_name in target_names:
                add_transfer(
                    target_name,
                    "shared_memory_to_register",
                    "target",
                    "local_load reloaded this value from shared memory into registers",
                    var_name,
                    ("loop" if var_name in defs_in_loop else "function"),
                )

    transfer_entries.sort(key=lambda entry: (int(entry["instruction_order"]), entry["occurrence_id"], entry["variable_name"]))
    artifact_entries: list[dict[str, str]] = []
    artifact_entries.append({"variable_name": "_artifacts", "transfer_type": "ptx_path", "role": "artifact", "detail": str(context["ptx_path"]), "occurrence_id": "", "execution_scope": "artifact", "instruction_order": "-1"})
    artifact_entries.append({"variable_name": "_artifacts", "transfer_type": "ttgir_path", "role": "artifact", "detail": str(context["ttgir_path"]), "occurrence_id": "", "execution_scope": "artifact", "instruction_order": "-1"})
    if context.get("ttir_path"):
        artifact_entries.append({"variable_name": "_artifacts", "transfer_type": "ttir_path", "role": "artifact", "detail": str(context["ttir_path"]), "occurrence_id": "", "execution_scope": "artifact", "instruction_order": "-1"})
    return transfer_entries + artifact_entries


def check_variable_transfer(
    ptx_path: str | None,
    *,
    input_shapes: Any | None = None,
    tune_args: Any | None = None,
) -> dict[str, object]:
    entries = find_all_variable_transfer(ptx_path, input_shapes=input_shapes, tune_args=tune_args)
    if not entries:
        return {}

    transfer_entries = [entry for entry in entries if entry.get("role") != "artifact"]
    artifact_map = {
        entry["transfer_type"]: entry["detail"]
        for entry in entries
        if entry.get("role") == "artifact"
    }
    return {
        "entries": transfer_entries,
        "_artifacts": artifact_map,
    }


def check_variable_storage(
    ptx_path: str | None,
    *,
    input_shapes: Any | None = None,
    tune_args: Any | None = None,
) -> dict[str, dict[str, str]]:
    transfer_summary = check_variable_transfer(ptx_path, input_shapes=input_shapes, tune_args=tune_args)
    if not transfer_summary:
        return {}

    storage: dict[str, dict[str, str]] = {}
    transfer_entries = transfer_summary.get("entries", [])
    grouped_pairs: dict[str, set[tuple[str, str]]] = {}
    if isinstance(transfer_entries, list):
        for transfer in transfer_entries:
            if not isinstance(transfer, dict):
                continue
            variable_name = str(transfer.get("variable_name", ""))
            if not variable_name:
                continue
            grouped_pairs.setdefault(variable_name, set()).add(
                (str(transfer.get("transfer_type", "")), str(transfer.get("role", "")))
            )
    for variable_name, transfer_pairs in grouped_pairs.items():
        if ("register_to_shared_memory", "source") in transfer_pairs:
            storage[variable_name] = {
                "storage": "register_to_shared_to_register",
                "reason": (
                    "discovered register_to_shared_memory transfer for this value"
                    " before tt.dot, so it is treated as register_to_shared_to_register"
                ),
            }
        elif ("global_to_register", "target") in transfer_pairs:
            storage[variable_name] = {
                "storage": "register_always",
                "reason": (
                    "discovered only a global_to_register transfer for this value"
                    " without shared-memory staging"
                ),
            }
        else:
            storage[variable_name] = {
                "storage": "other",
                "reason": "the tt.load result was found, but its later transfer sequence could not be resolved from TTGIR",
            }
    storage["_artifacts"] = dict(transfer_summary.get("_artifacts", {}))  # type: ignore[arg-type]
    return storage


def analyze_dot_forced_shared_reload(ptx_path: str | None) -> list[dict[str, str | bool]]:
    context = _load_ttgir_context(ptx_path)
    if not context:
        return []

    defs = dict(context["defs"])
    load_variables = list(context["load_variables"])
    resolve_loc_name = context["resolve_loc_name"]
    dot_ops = list(context["dot_ops"])
    load_name_map = {
        f"%{raw_variable}": (resolve_loc_name(loc_id) or raw_variable)  # type: ignore[operator]
        for raw_variable, loc_id in load_variables
    }

    def trace_load_origin(source: str, seen: set[str] | None = None) -> str | None:
        if source in load_name_map:
            return source
        if seen is None:
            seen = set()
        if source in seen:
            return None
        seen.add(source)
        rhs = defs.get(source, "")
        for dependency in re.findall(_TTGIR_VALUE_RE, rhs):
            found = trace_load_origin(dependency, seen)
            if found is not None:
                return found
        return None

    def path_has_local_alloc(source: str, seen: set[str] | None = None) -> bool:
        if seen is None:
            seen = set()
        if source in seen:
            return False
        seen.add(source)
        rhs = defs.get(source, "")
        if "local_alloc" in rhs:
            return True
        for dependency in re.findall(_TTGIR_VALUE_RE, rhs):
            if path_has_local_alloc(dependency, seen):
                return True
        return False

    def is_passthrough_op(rhs: str) -> bool:
        return rhs.startswith(
            (
                "triton_gpu.convert_layout ",
                "ttg.convert_layout ",
                "tt.trans ",
                "ttg.memdesc_trans ",
                "arith.extf ",
                "arith.truncf ",
                "arith.bitcast ",
                "arith.fpext ",
                "arith.fptrunc ",
                "arith.sitofp ",
                "arith.uitofp ",
                "arith.index_cast ",
            )
        )

    def find_final_shared_reload(source: str, seen: set[str] | None = None) -> str | None:
        if seen is None:
            seen = set()
        if source in seen:
            return None
        seen.add(source)
        rhs = defs.get(source, "")
        if "local_load" in rhs:
            return source
        if not is_passthrough_op(rhs):
            return None
        dependencies = re.findall(_TTGIR_VALUE_RE, rhs)
        if len(dependencies) != 1:
            return None
        return find_final_shared_reload(dependencies[0], seen)

    def operand_origin_kind(source: str) -> str:
        rhs = defs.get(source, "")
        if rhs.startswith("tt.load "):
            return "direct_tt_load"
        if trace_load_origin(source) is not None:
            return "derived_from_tt_load"
        if rhs:
            return "computed"
        return "computed"

    results: list[dict[str, str | bool]] = []
    for dot_result, lhs, rhs in dot_ops:
        for side, operand in (("lhs", lhs), ("rhs", rhs)):
            load_origin = trace_load_origin(operand)
            final_local_load = find_final_shared_reload(operand)
            forced = final_local_load is not None and path_has_local_alloc(final_local_load)
            if forced:
                classification = "register_to_shared_to_register"
                if load_origin is None:
                    reason = (
                        "operand was produced by intermediate ops, then staged through shared memory "
                        "and reloaded before tt.dot"
                    )
                else:
                    reason = "operand traces from tt.load through local_alloc/local_load before tt.dot"
            elif load_origin is not None:
                classification = "register_always"
                reason = "operand traces from tt.load into tt.dot without a visible shared-memory reload path"
            else:
                classification = "other"
                reason = "operand reaches tt.dot through intermediate ops without a visible shared-memory reload path"
            results.append(
                {
                    "dot_result": dot_result,
                    "side": side,
                    "operand": operand,
                    "load_origin": (load_origin or ""),
                    "variable_name": (load_name_map[load_origin] if load_origin is not None else ""),
                    "origin_kind": operand_origin_kind(operand),
                    "direct_tt_load_operand": defs.get(operand, "").startswith("tt.load "),
                    "from_tt_load": load_origin is not None,
                    "forced_register_to_shared_to_register": forced,
                    "classification": classification,
                    "reason": reason,
                }
            )
    return results

def _resource_usage_summary(
    name: str,
    launch_context: Mapping[str, Any] | None,
    tune_args: Any | None = None,
    input_shapes: Any | None = None,
) -> str:
    launch_context = launch_context or {}
    shared_used = int(
        launch_context.get(
            "compiled_shared_memory_bytes_per_block",
            launch_context.get("attempted_shared_memory_bytes_per_block", 0),
        )
    )
    shared_total = int(launch_context.get("available_shared_memory_bytes_per_sm", 0))
    register_file_used = int(launch_context.get("attempted_register_file_size_bytes_per_block", 0))
    register_file_total = int(launch_context.get("available_register_file_size_bytes_per_sm", 0))
    registers_used = int(launch_context.get("attempted_registers_per_block", 0))
    registers_total = int(launch_context.get("available_registers_per_sm", 0))
    summary = (
        f"{name} tuner resources: \n"
        f"shared={shared_used}/{shared_total} bytes per block/SM, \n"
        f"register_file={register_file_used}/{register_file_total} bytes per block/SM, \n"
        f"registers={registers_used}/{registers_total} per block/SM "
    )
    transfers = check_variable_transfer(
        launch_context.get("compiled_ptx_path"),
        input_shapes=input_shapes,
        tune_args=tune_args,
    )
    if transfers:
        transfer_fields = []
        details = transfers.get("entries", [])
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                if not all(key in detail for key in ("variable_name", "transfer_type", "role")):
                    continue
                transfer_fields.append(
                    f"\n{detail['variable_name']}={detail['transfer_type']}[{detail['role']}]"
                )
        if transfer_fields:
            summary += f", \ntransfers={', '.join(transfer_fields)}"
        artifact_entry = transfers.get("_artifacts", {})
        if isinstance(artifact_entry, Mapping) and artifact_entry.get("ptx_path"):
            summary += f", \nptx={artifact_entry['ptx_path']}"
    formatted_tune_args = _format_tune_args(tune_args)
    if formatted_tune_args:
        summary += f", \ntune_args={formatted_tune_args}"
    formatted_input_shapes = _format_input_shapes(input_shapes)
    if formatted_input_shapes:
        summary += f", \ninput_shapes={formatted_input_shapes}"
    return summary


def _resource_failure_text(prefix: str, exc: "OutOfResourcesWithDetail") -> str:
    launch_context = getattr(exc, "launch_context", {}) or {}
    return f"{prefix}: {exc} [{_resource_usage_summary('failed', launch_context)}]"




class OutOfResourcesWithDetail(OutOfResources):
    """Richer Triton resource error with kernel and block-launch context."""

    def __init__(
        self,
        required: int,
        limit: int,
        name: str,
        *,
        kernel_name: str | None = None,
        launch_context: Mapping[str, Any] | None = None,
        extra_detail: str | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        super().__init__(required, limit, name)
        self.kernel_name = kernel_name
        self.launch_context = dict(launch_context or {})
        self.extra_detail = extra_detail
        self.original_exception = original_exception

    @classmethod
    def from_out_of_resources(
        cls,
        exc: OutOfResources,
        *,
        kernel_name: str | None = None,
        launch_context: Mapping[str, Any] | None = None,
        extra_detail: str | None = None,
    ) -> "OutOfResourcesWithDetail":
        return cls(
            exc.required,
            exc.limit,
            exc.name,
            kernel_name=kernel_name,
            launch_context=launch_context,
            extra_detail=extra_detail,
            original_exception=exc,
        )

    def _resource_hint(self) -> str:
        resource = str(self.name).lower()
        if "register" in resource:
            return "The block likely needs more register file capacity per SM than the hardware can provide."
        if "shared" in resource:
            return "The block likely needs more shared memory per SM than the hardware can provide."
        return "The block likely exceeds a per-SM hardware resource limit for this launch configuration."

    def _format_context(self) -> str | None:
        if not self.launch_context:
            return None
        parts = [f"{key}={value}" for key, value in self.launch_context.items()]
        return "\n".join(parts)

    def __str__(self) -> str:
        parts = [super().__str__(), self._resource_hint()]
        if self.kernel_name:
            parts.append(f"Kernel: {self.kernel_name}.")
        context = self._format_context()
        if context:
            parts.append(f"Launch context: {context}.")
        if self.extra_detail:
            parts.append(self.extra_detail.rstrip(".") + ".")
        return " ".join(parts)


def get_sm_resource_limits() -> dict[str, Any]:
    device = triton.runtime.driver.active.get_current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    torch_props = torch.cuda.get_device_properties(device)
    max_num_regs = props["max_num_regs"]
    warp_size = props["warpSize"]
    max_threads_per_block = int(getattr(torch_props, "max_threads_per_block", 1024))
    return {
        "available_shared_memory_bytes_per_sm": props["max_shared_mem"],
        "available_registers_per_sm": max_num_regs,
        "available_register_file_size_bytes_per_sm": max_num_regs * 4,
        "warp_size": warp_size,
        "max_threads_per_block": max_threads_per_block,
        "max_warps_per_block": max(1, max_threads_per_block // max(1, warp_size)),
        "multiprocessor_count": props["multiprocessor_count"],
    }