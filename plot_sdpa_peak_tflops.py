from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    gpus = ["T4", "A10"]
    peak_tflops = [23, 72]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(gpus, peak_tflops, color=["#4C78A8", "#F58518"], width=0.55)

    ax.set_title("SDPA GPU Benchmark")
    ax.set_xlabel("GPU")
    ax.set_ylabel("Peak Throughput (TFLOPs)")
    ax.set_ylim(0, max(peak_tflops) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, peak_tflops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value} TFLOPs",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()

    output_path = Path("imgs/sdpa_peak_tflops_bar.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(output_path)


if __name__ == "__main__":
    main()
