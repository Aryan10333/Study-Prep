# 07. Inference Engines: Framework Architectures and Capabilities

Deploying LLMs in production requires selecting the appropriate inference serving framework. Serving engines manage memory allocation, request scheduling, batching execution, and hardware kernel mapping. Each stack targets specific hardware constraints and operational SLAs.

---

## 1. Deep-Dive Architecture Breakdown

### vLLM
vLLM is an open-source, high-throughput LLM serving engine. It introduced **PagedAttention** to eliminate VRAM memory fragmentation.
- **Core Features**: Logical/physical block KV Cache manager, continuous batching scheduler, block hash prefix caching, ray-based tensor parallelism, and broad model coverage.
- **Optimal Environment**: Multi-GPU cluster serving (A100/H100/L4) for generic APIs with high concurrency.

### SGLang (Structured Generation Language)
SGLang focuses on optimizing fast execution for complex prompting, structured generation (JSON/schema), and multi-agent workflows.
- **Core Features**: **RadixAttention** prefix caching (radix tree representation), **XGrammar** compiled structured decoding masks, and high-performance pipeline parallelism for Mixture-of-Experts (MoE) architectures (e.g. Mixtral/DeepSeek).
- **Optimal Environment**: Multi-turn agent loops, structured extraction pipelines, and models with massive system prompts.

### TensorRT-LLM
NVIDIA's proprietary, high-performance compiler and engine stack for LLM serving.
- **Core Features**: Ahead-of-time (AOT) CUDA graph compilation, custom NVIDIA Tensor Core kernels, highly optimized FP8 execution, and in-flight batching.
- **Trade-Off**: Requires a separate compilation step to build engine binaries (`trtllm-build`), leading to high compilation times and strict hardware binding (the compiled engine only runs on the exact GPU model used for compilation).
- **Optimal Environment**: High-volume, enterprise single-model serving on NVIDIA Hopper (H100/H200) clusters where minimizing TPOT is the primary constraint.

### llama.cpp & Ollama
A C++ implementation of transformer architectures optimized for local consumer hardware.
- **Core Features**: Block-wise quantization formats (GGUF), CPU/GPU heterogeneous memory-mapped execution, Metal API support (Apple Silicon), and zero external dependencies.
- **Optimal Environment**: Edge serving, developer local test setups, and CPU-only cloud instances.

### Text Generation Inference (TGI)
Created by Hugging Face, TGI was one of the early production-grade engines. While it introduced key features like continuous batching and streaming tokens, it has transitioned to maintenance mode as open-source frameworks like vLLM and SGLang have overtaken its performance.

---

## 2. Comprehensive Engine Comparison Matrix

The table below compares serving engines across operational dimensions:

| Feature Dimension | vLLM | SGLang | TensorRT-LLM | llama.cpp / Ollama |
|---|---|---|---|---|
| **Primary Bottleneck** | Memory Bandwidth | Scheduling Overheads | Network Bandwidth | Hardware Access Rate |
| **Prefix Caching** | Block-Level Hashing | Radix Tree (RadixAttention) | None (Static) | Local context caching |
| **Structured Output** | RegEx/JSON Outlines | Compiled XGrammar | Simple JSON Schema | Token-level constraints |
| **Quantization Support** | FP8, AWQ, GPTQ, INT8 | FP8, AWQ, GPTQ, INT8 | FP8, INT8/INT4 (SmoothQuant) | GGUF (K-Quant) |
| **Compilation Overhead** | Zero (JIT Python/C++) | Zero (JIT C++) | High (Pre-compile required) | Zero (Direct binary run) |
| **Multi-GPU TP/PP** | Native (Ray) | Native (Pytorch/Ray) | Custom MPI/NCCL | CPU/GPU load splits |
| **Preferred Hardware** | H100, A100, L4 | H100, A100, L4 | H100, H200 | Apple Silicon, CPUs, Edge |

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
