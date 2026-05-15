# Project Intake And Repository Map

## 1. Project Identity Card

| Field | Value | Evidence | Confidence |
|---|---|---|---|
| Project Name | SGLang | README.md:1, pyproject.toml:6 | High |
| Owner | sgl-project (LMSYS) | README.md:81, GitHub | High |
| Repository URL | https://github.com/sgl-project/sglang | README.md:169 | High |
| Local Path | d:\codeStudy\sglang | Filesystem | High |
| License | Apache 2.0 | LICENSE:1-3 | High |
| Version | Dynamic (setuptools_scm based on git tag) | pyproject.toml:7, version.py:13-20 | High |
| Latest Release | v0.5 (inferred from changelogs/PRs) | README news mentions v0.2-0.4 | Medium |
| Python Requirement | >=3.10 | pyproject.toml:10 | High |
| Total Commits | 12,649 | `git rev-list --count HEAD` | High |
| Latest Commit | 2026-05-15: Enable SGLANG_OPT_FP8_WO_A_GEMM by default | `git log -1` | High |
| Primary Language | Python (with Rust for gRPC, C++ for CUDA kernels) | pyproject.toml, 3rdparty/, rust/, sgl-kernel/ | High |
| Total Python Files | 1,976 in main package | `find . -name '*.py' \| wc -l` | High |

## 2. Problem Domain And Target Users

### Problem Domain
SGLang is a **high-performance inference serving framework** for Large Language Models (LLMs), Vision Language Models (VLMs), and Diffusion Models. It solves the problem of serving generative AI models at scale with low latency and high throughput across diverse hardware platforms.

### Target Users
| User Type | Use Case | Evidence |
|---|---|---|
| AI/ML Engineers | Deploy LLMs for production inference | README:58-66 |
| Research Labs (MIT, Stanford, Tsinghua) | Model evaluation, training rollouts | README Adoption section |
| Cloud/Enterprise (xAI, LinkedIn, AMD, NVIDIA) | Large-scale production serving (400,000+ GPUs) | README:79-80 |
| Post-training Frameworks (verl, AReaL, Miles) | RL training backbone | README:66 |
| Open-source Contributors | Kernel optimization, model support | 12,649 commits, active CI |
| Hardware Vendors (NVIDIA, AMD, Intel, Google TPU) | Hardware enablement | README:64 |

### Core Capabilities (READ-verified)
| Capability | Description | Source |
|---|---|---|
| RadixAttention | Prefix caching for repeated prompt patterns | README:62 |
| Zero-overhead CPU Scheduler | Batch scheduling without GPU overhead | README:62 |
| Prefill-Decode Disaggregation | Separate prefill and decode phases across GPUs | README:62 |
| Speculative Decoding | Draft-then-verify for faster generation | README:62 |
| Continuous Batching | Dynamic request batching | README:62 |
| PagedAttention | KV-cache memory management | README:62 |
| Tensor/Pipeline/Expert/Data Parallelism | Distributed inference | README:62 |
| Structured Outputs | JSON/Regex/Grammar-constrained generation | README:62 |
| Quantization (FP4/FP8/INT4/AWQ/GPTQ) | Compressed model serving | README:62 |
| Multi-LoRA Batching | Serve multiple LoRA adapters | README:62 |
| Broad Model Support | Llama, Qwen, DeepSeek, GPT, Gemma, Mistral, etc. | python/sglang/srt/models/ (191 model files) |
| Hardware Support | NVIDIA, AMD, Intel CPU, Google TPU, Ascend NPU | README:64 |
| Diffusion Models | WAN, Qwen-Image generation | README:63 |

## 3. Project Type Classification

| Candidate Type | Evidence | Confidence | Analysis Focus |
|---|---|---|---|
| **AI Agent/LLM (primary)** | LLM serving framework, RadixAttention, speculative decoding, continuous batching | High | Tool safety, prompt/memory, eval |
| **Web Backend** | FastAPI HTTP server, OpenAI-compatible API, gRPC server | High | Routes, middleware, auth, services |
| **CLI** | `sglang serve`, `sglang generate`, `sglang version` via argparse | High | Commands, flags, config, exit codes |
| **SDK/Library** | Public Python API (`sglang.lang` frontend), pip package | High | Public exports, types, compatibility |
| **Infrastructure** | GPU kernel optimization (sgl-kernel), CUDA/CUTLASS, distributed systems | High | Resource model, provider, scaling |
| **DevOps** | Docker, Kubernetes, CI/CD, Observability (Prometheus) | Medium | Deployment patterns, health check |

## 4. Technology Stack Fingerprint

| Layer | Technology | Version | Evidence | Risk |
|---|---|---|---|---|
| **Language** | Python | >=3.10 | pyproject.toml:10 | Low |
| **DL Framework** | PyTorch | 2.11.0 | pyproject.toml:70 | Medium (tightly coupled) |
| **ML Backend** | Transformers (HuggingFace) | 5.6.0 | pyproject.toml:78 | Medium |
| **CUDA/GPU** | flashinfer, flash-attn-4, triton, nvidia-cutlass-dsl | 0.6.11, 4.0.0, -, 4.5.0 | pyproject.toml:32-33,61,40 | Medium (hardware-locked) |
| **HTTP Server** | FastAPI + uvicorn/uvloop | - | pyproject.toml:29,79,80 | Low |
| **HTTP/2 Server** | granian (optional) | >=2.6.0 | pyproject.toml:128-129 | Low |
| **gRPC** | Rust-based via PyO3 + smg-grpc-servicer | >=0.5.0 | pyproject.toml:83, rust/ | Medium (native extension) |
| **Structured Output** | xgrammar, outlines, llguidance, interegular | 0.2.0, 0.1.11, >=0.7.11, - | pyproject.toml:82,46,34,33 | Low |
| **Serialization** | orjson, msgspec, pydantic | - | pyproject.toml:44,36,53 | Low |
| **Observability** | prometheus-client, py-spy, OpenTelemetry (optional) | >=0.20.0, -, - | pyproject.toml:49,51,121-125 | Low |
| **Custom Kernels** | sglang-kernel, sgl-deep-gemm, tokenspeed_mla, quack-kernels, tilelang | 0.4.2, 0.1.0, 0.1.1, >=0.4.1, 0.1.8 | pyproject.toml:62-65 | Medium |
| **Model Hub** | modelscope, blobfile, runai-model-streamer | - | pyproject.toml:35,22,94-95 | Low |
| **Security** | ssl_utils.py exists; no dedicated auth module found | python/sglang/srt/entrypoints/ssl_utils.py | Medium (basic) |
| **Distributed** | Ray (optional), ZeroMQ, NCCL (via PyTorch) | >=2.54.0, >=25.1.2 | pyproject.toml:117,55 | Medium |
| **Build System** | setuptools + setuptools-rust + setuptools-scm | - | pyproject.toml:2 | Low |
| **Package Manager** | uv (implied by [tool.uv]) | - | pyproject.toml:87,164 | Low |

## 5. License And Community Signals

| Signal | Value | Evidence | Confidence |
|---|---|---|---|
| License Type | Apache 2.0 | LICENSE:1-3 | High |
| Commercial Use | Allowed (explicit grant) | Apache 2.0 Section 2 | High |
| Patent Grant | Yes | Apache 2.0 Section 3 | High |
| Copyleft | No (permissive) | Apache 2.0 does not require derivative opensource | High |
| Attribution Required | Yes (NOTICE file) | Apache 2.0 Section 4d | High |
| LMSYS Non-profit Host | Yes | README:81 | High |
| a16z Grant Recipient | Yes (third batch) | README:40 | High |
| PyTorch Ecosystem Member | Yes | README:44 | High |
| Enterprise Adoption | xAI, AMD, NVIDIA, LinkedIn, Cursor, Oracle, Azure, AWS, etc. | README:79 | Medium |
| GPU Deployments | 400,000+ claimed | README:79 | Low (marketing, unverified) |
| Stars/Forks | Not queried (no network) | N/A | Unverified |
| Issue Resolution | Active (badges in README) | README:7-8 | Medium |
| Documentation | docs.sglang.io, deepwiki.com | README:17,9 | High |
| Community Channels | Slack, Weekly Dev Meeting | README:19-20 | High |

## 6. Annotated Repository Tree

```
sglang/  (root: d:\codeStudy\sglang)
├── README.md                     # Project overview, features, adoption
├── LICENSE                       # Apache 2.0
├── AGENTS.md                     # Agent instructions (Claude Code)
├── GEMINI.md                     # Gemini instructions
├── .claude/                      # Claude Code project config + rules
│   └── rules/
│       └── speculative-naming.md # Naming conventions for spec decoding
├── .github/                      # CI/CD workflows (50+ workflow files)
│   ├── workflows/                # pr-test, nightly-test, lint, etc.
│   └── CODEOWNERS                # Code ownership
├── .devcontainer/                # VS Code dev container setup
├── .pre-commit-config.yaml       # Pre-commit hooks (lint, format)
├── .codespellrc                  # Spell check config
├── .coveragerc                   # Coverage config
├── .isort.cfg                    # Import sorting config
│
├── python/                       # ★ MAIN SOURCE CODE (Python)
│   ├── pyproject.toml            # Build config, dependencies
│   ├── pyproject_cpu.toml        # CPU-only variant
│   ├── pyproject_npu.toml        # Ascend NPU variant
│   ├── pyproject_xpu.toml        # Intel XPU variant
│   └── sglang/                   # Main package
│       ├── __init__.py
│       ├── version.py            # Dynamic version resolution
│       ├── launch_server.py      # Entry: run_server()
│       ├── cli/                  # CLI commands (serve, generate, version)
│       ├── srt/                  # ★★ SGLang Runtime (CORE)    
│       │   ├── server_args.py    #     ~7950 lines of server config
│       │   ├── environ.py        #     Environment variable management
│       │   ├── entrypoints/      #     HTTP/gRPC/Engine servers
│       │   ├── managers/         #     Scheduler, tokenizer, cache, TP workers
│       │   ├── models/           #     191 model implementations
│       │   ├── model_executor/   #     Model execution engine
│       │   ├── model_loader/     #     Model weight loading
│       │   ├── layers/           #     Attention, MoE, quantization layers
│       │   ├── mem_cache/        #     Memory/cache management
│       │   ├── constrained/      #     Structured output (grammar)
│       │   ├── speculative/      #     Speculative decoding
│       │   ├── disaggregation/   #     Prefill-decode disaggregation
│       │   ├── distributed/      #     Distributed inference utils
│       │   ├── lora/             #     LoRA adapter support
│       │   ├── multimodal/       #     Multi-modal processing
│       │   ├── sampler/          #     Sampling strategies
│       │   ├── grpc/             #     gRPC protocol definitions
│       │   ├── parser/           #     Reasoning/tool call parsers
│       │   ├── connector/        #     Model connector (HF, etc.)
│       │   ├── configs/          #     Model-specific configs
│       │   ├── compilation/      #     torch.compile integration
│       │   ├── hardware_backend/ #     Hardware abstraction
│       │   ├── observability/    #     Metrics, logging
│       │   ├── multiplex/        #     Multi-model serving
│       │   ├── session/          #     Session management
│       │   ├── eplb/             #     Expert parallel load balancing
│       │   ├── elastic_ep/       #     Elastic expert parallelism
│       │   ├── dllm/             #     Diffusion LLM support
│       │   ├── batch_invariant_ops/  # Batch-invariant operations
│       │   └── utils/            #     Utilities
│       ├── lang/                 # SGLang frontend DSL
│       │   ├── api.py            #     Public API
│       │   ├── ir.py             #     Intermediate representation
│       │   ├── backend/          #     Backend runtime integration
│       │   ├── interpreter.py    #     Program interpreter
│       │   └── tracer.py         #     Execution tracer
│       ├── jit_kernel/           # JIT-compiled kernels
│       ├── multimodal_gen/       # Diffusion/generation models
│       ├── eval/                 # Evaluation infrastructure
│       ├── benchmark/            # Benchmark scripts
│       ├── auto_benchmark.py     # Automated benchmarking
│       ├── bench_offline_throughput.py  # Offline throughput bench
│       ├── bench_one_batch.py    # Single batch benchmark
│       ├── bench_serving.py      # Online serving benchmark
│       ├── profiler.py           # Performance profiler
│       ├── check_env.py          # Environment checker
│       ├── global_config.py      # Global configuration
│       ├── kernel_api_logging.py # Kernel API logging
│       └── utils.py              # General utilities
│
├── test/                         # Test suite
│   ├── srt/                      # Runtime tests (mirrors sglang.srt)
│   ├── manual/                   # Manual test scripts
│   ├── registered/               # Registered test cases
│   ├── lm_eval_configs/          # LM evaluation configs
│   └── run_suite.py              # Test suite runner
│
├── sgl-kernel/                   # ★ CUDA/C++ kernel library (separate package)
│   ├── pyproject.toml            # Kernel package config
│   ├── src/                      # Kernel source (C++/CUDA)
│   └── include/                  # Kernel headers
│
├── sgl-model-gateway/            # Model gateway service (separate service)
│   ├── bindings/python/          # Python bindings
│   └── e2e_test/                 # End-to-end tests
│
├── rust/                         # Rust components (gRPC server)
│   └── sglang-grpc/              # gRPC implementation in Rust
│
├── 3rdparty/                     # Third-party kernels (AMD etc.)
├── docker/                       # Docker configurations
├── docs/ + docs_new/             # Documentation (Sphinx + new docs)
├── examples/                     # Usage examples
├── scripts/                      # Utility scripts
├── benchmark/                    # Additional benchmarks
├── assets/                       # Static assets (logo, images)
└── proto/                        # Protobuf definitions
```

### Size Metrics
| Metric | Count | Evidence |
|---|---|---|
| Python files in sglang package | 1,976 | `find . -name '*.py' \| wc -l` |
| Model implementations | 191 | `ls python/sglang/srt/models/ \| wc -l` |
| CI workflow files | 50+ | `ls .github/workflows/` |
| Lines in server_args.py | 7,950 | `wc -l server_args.py` |
| Git commits | 12,649 | `git rev-list --count HEAD` |
| Top-level directories | 16 | `ls -d */` |

## 7. Entry Point Map

| Entry Type | Path | Symbol/Command | Purpose | Evidence |
|---|---|---|---|---|
| CLI - Serve | python/sglang/cli/main.py:12-21 | `sglang serve` | Launch inference server | pyproject.toml:173, cli/main.py:37-39 |
| CLI - Generate | python/sglang/cli/main.py:22-26 | `sglang generate` | Run multimodal inference | pyproject.toml:173, cli/main.py:22-26 |
| CLI - Version | python/sglang/cli/main.py:27-33 | `sglang version` | Show version info | pyproject.toml:173, cli/main.py:7-9 |
| CLI - Kill All | python/sglang/cli/killall.py | `killall_sglang` | Kill all SGLang processes | pyproject.toml:174 |
| HTTP Server | python/sglang/srt/entrypoints/http_server.py | `launch_server(server_args)` | Main HTTP API server (FastAPI) | launch_server.py:48-50 |
| gRPC Server | python/sglang/srt/entrypoints/grpc_server.py | `serve_grpc(server_args)` | gRPC API server | launch_server.py:33-35 |
| Ray Server | python/sglang/srt/ray/http_server.py | `launch_server(server_args)` | Ray-distributed server | launch_server.py:38-45 |
| Engine Bootstrap | python/sglang/srt/entrypoints/engine.py | `Engine` class | Core inference engine | srt/entrypoints/engine.py |
| Frontend DSL | python/sglang/lang/api.py | `sglang.lang` module | High-level programming interface | sglang/lang/api.py |
| Environment Check | python/sglang/check_env.py | `check_env` module | Verify environment setup | sglang/check_env.py |
| Benchmark | python/sglang/bench_serving.py | `bench_serving` module | Online serving benchmark | sglang/bench_serving.py |

## 8. Configuration And Manifest Map

| File | Purpose | Key Settings | Evidence |
|---|---|---|---|
| python/pyproject.toml | Main package build & deps | All dependencies, entry points, build config | pyproject.toml |
| python/pyproject_cpu.toml | CPU-only build variant | CPU-specific dependencies | File exists |
| python/pyproject_npu.toml | Ascend NPU build variant | NPU-specific dependencies | File exists |
| python/pyproject_xpu.toml | Intel XPU build variant | XPU-specific dependencies | File exists |
| python/sglang/srt/server_args.py | Server configuration (~7950 lines) | All CLI arguments, model config, parallelism settings | server_args.py:1 |
| python/sglang/srt/environ.py | Environment variable management | SGLANG_* environment variables | environ.py:38-59 |
| sgl-kernel/pyproject.toml | CUDA kernel package build | Kernel compilation settings | File exists |
| .pre-commit-config.yaml | Code quality pre-commit hooks | isort, black, ruff, etc. | .pre-commit-config.yaml |
| .github/workflows/ | CI/CD pipeline definitions | Lint, test, nightly, release | Directory listing |
| docker/ | Docker build files | Container configuration | Directory listing |
| .devcontainer/ | Dev container setup | VS Code remote config | Directory listing |
| .claude/rules/ | Claude Code project rules | Speculative decoding naming conventions | .claude/rules/speculative-naming.md |
| AGENTS.md | Agent automation instructions | Subagent workflow rules | AGENTS.md |
| GEMINI.md | Gemini-specific instructions | Gemini adapter config | GEMINI.md |

## 9. Docs vs Code Consistency Check

| Claim In Docs | Code Evidence | Status | Notes |
|---|---|---|---|
| "Fast Runtime with RadixAttention" | srt/mem_cache/ contains memory pool, radix cache implementations | Consistent | Multiple mem_cache modules found |
| "Speculative Decoding" | srt/speculative/ directory exists with Eagle, NGram implementations | Consistent | Full speculative decoding pipeline |
| "Continuous Batching" | srt/managers/schedule_batch.py, scheduler.py | Consistent | Core scheduler implementation |
| "PagedAttention" | srt/layers/attention/ contains paged attention implementations | Consistent | CUDA/Triton kernel backends |
| "191 model implementations" | srt/models/ has 191 Python files | Consistent | Covers all major arch families |
| "NVIDIA, AMD, Intel, TPU, NPU support" | srt/hardware_backend/, platforms/, pyproject variants | Consistent | Hardware-specific paths exist |
| "Prefill-Decode Disaggregation" | srt/disaggregation/ directory | Consistent | Separate encode/decode servers |
| "OpenAI API Compatible" | srt/entrypoints/openai/ directory | Consistent | OpenAI protocol adapters |
| "Structured Outputs" | srt/constrained/ with xgrammar, outlines, llguidance backends | Consistent | Multiple grammar backends |
| "Quantization (FP4/FP8/INT4/AWQ/GPTQ)" | srt/layers/quantization/ likely path | Consistent with dependency list | torchao, compressed-tensors in deps |
| "400,000+ GPUs deployed" (marketing) | Not verifiable from code | Unverifiable | Marketing claim |
| "Trillions of tokens daily" (marketing) | Not verifiable from code | Unverifiable | Marketing claim |

## 10. Initial Research Risks And Open Questions

| Risk / Question | Severity | Notes |
|---|---|---|
| **server_args.py at 7950 lines** is a God configuration object — risk of config spaghetti | High | Needs architecture analysis in Stage 2 |
| **Tight coupling to PyTorch 2.11.0** — version lock-in | Medium | Requires compatible ecosystem |
| **CUDA dependency is pervasive** — CPU-only mode may be limited | High | Need to verify CPU/TPU paths |
| **Rust gRPC component** adds build complexity | Medium | PyO3 binding, Cargo dependency |
| **No apparent auth module** — likely relies on network-level security | Medium | Need to verify in Stage 9 |
| **sgl-kernel separate compilation** — build failure risk | Medium | C++/CUDA compilation required |
| **Marketing claims (400K GPUs)** cannot be verified | Low | Treat as directional, not factual |
| **191 model files** — how do they share code vs copy-paste? | Medium | Architecture analysis needed |
| **CLI only has `serve`, `generate`, `version`** — limited CLI surface | Low | Core functionality via HTTP API |
| **Multiple pyproject variants** — how well tested is each platform path? | Medium | CI shows nightly testing |
| **AGENTS.md and GEMINI.md** in root — agent-driven development workflow | Low | Novel integration pattern |

## 11. Top 10 Priority Files For Further Analysis

| Priority | File | Reason |
|---|---|---|
| 1 | python/sglang/srt/entrypoints/engine.py | Core inference engine — central orchestrator |
| 2 | python/sglang/srt/managers/scheduler.py | Batch scheduler — key innovation |
| 3 | python/sglang/srt/managers/tp_worker.py | Tensor parallelism worker — distributed execution |
| 4 | python/sglang/srt/mem_cache/ | Memory management — RadixAttention, paged attention |
| 5 | python/sglang/srt/model_executor/ | Model execution path — forward pass |
| 6 | python/sglang/srt/server_args.py | All configuration — understand the config universe |
| 7 | python/sglang/srt/speculative/ | Speculative decoding — performance innovation |
| 8 | python/sglang/srt/disaggregation/ | Prefill-decode split — distributed architecture |
| 9 | python/sglang/srt/layers/attention/ | Attention kernels — compute hot path |
| 10 | python/sglang/srt/constrained/ | Structured output — grammar-constrained generation |
