# flash-attn-economical-gpu
A set of implementations on Triton based FlashAttention are proposed, which beats or reaches comparable performance with popular library such as Pytorch's FlashAttention SDPA, across multiple economical GPU architectures such as Turing and Ampere.

## Results

![MMA on Turing](imgs/turing/mma_different_data_length.png)

Our Triton MMA matrix multiplication kernel outperforms `torch.matmul` on Turing across the tested data lengths, reaching up to 80% higher performance.

![MMA on Ampere](imgs/ampere/mma_kernel_pipeline_runtime_tflops.png)

The Triton MMA matrix multiplication kernel also improves over `torch.matmul` on Ampere, with gains up to 25% in the measured configurations.

![SDPA on Turing](imgs/turing/flash_atten_kernel_pipeline_runtime_tflops_128.png)

For `D=128` on Turing, the basic Triton FlashAttention kernel substantially exceeds PyTorch math and efficient SDPA backends, while the optimized kernel adds about 8% more throughput.

![SDPA on Turing for D=128](imgs/turing/flash_atten_kernel_pipeline_runtime_tflops_128_k1_2.png)

Across different sequence lengths with `D=128` on Turing, the optimized Triton SDPA kernel remains the fastest tested backend.

![SDPA on Turing for D=256](imgs/turing/flash_atten_kernel_pipeline_runtime_tflops_256_k1_2.png)

For `D=256` on Turing, the optimized Triton SDPA kernel continues to lead the tested PyTorch SDPA alternatives.

![SDPA on Ampere](imgs/ampere/flash_atten_kernel_pipeline_runtime_tflops_128.png)

On Ampere, the sliced Triton SDPA kernel reaches performance comparable to PyTorch FlashAttention, which is backed by cuDNN.

![SDPA on Ampere for D=128](imgs/ampere/flash_atten_kernel_pipeline_runtime_tflops_128_k_1_2_3.png)

For `D=128` on Ampere, the sliced Triton SDPA kernel stays close to PyTorch FlashAttention as problem size increases.

![SDPA on Ampere for D=256](imgs/ampere/flash_atten_kernel_pipeline_runtime_tflops_256_k_1_2_3.png)

For `D=256` on Ampere, the sliced Triton SDPA kernel reaches nearly 60 TFLOPs and is at most about 16% slower than PyTorch FlashAttention.

![GPU Benchmark on SDPA](imgs/sdpa_peak_tflops_bar.png)

The SDPA peak-throughput benchmark shows the A10 delivering at least 3x the throughput of the T4.

![TTFT on Qwen2.5 and Qwen3.5](imgs/ttft/20260504T111251_885343Z_llm_ttft.png)

In LLM TTFT tests on Qwen2.5 and Qwen3.5, the sliced Triton kernel remains close to PyTorch FlashAttention and clearly faster than PyTorch efficient SDPA.
