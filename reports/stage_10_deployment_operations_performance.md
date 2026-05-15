# Deployment, Operations And Performance

## 1. Local Startup

| Method | Command | Prerequisites | Verified | Evidence |
|---|---|---|---|---|
| **CLI serve** | `sglang serve --model-path <model> --port 30000` | CUDA GPU, PyTorch, flashinfer | ❌ Not executed (no GPU) | cli/serve.py, server_args.py |
| **CLI generate** | `sglang generate --model-path <model> --prompt "Hello"` | Same as serve | ❌ Not executed | cli/generate.py |
| **Python Engine** | `from sglang import Engine; engine = Engine(model_path=...)` | Same as serve | ❌ Not executed | engine.py:174 |
| **Docker (NVIDIA)** | `docker run --gpus all lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path ...` | Docker + nvidia-container-toolkit | ❌ Not executed | docker/Dockerfile |
| **Docker Compose (monitoring)** | `docker compose -f examples/monitoring/docker-compose.yaml up` | Docker | ❌ Not executed | examples/monitoring/docker-compose.yaml |
| **Devcontainer** | VS Code + `.devcontainer/devcontainer.json` | VS Code Dev Containers extension | ❌ Not executed | .devcontainer/ |
| **Environment check** | `python -m sglang.check_env` | Python 3.10+ | ❌ Not executed | check_env.py |

### Startup Time Profile (Static Analysis)

| Phase | What Happens | Files | Estimated % of Startup |
|---|---|---|---|
| **1. Config parsing** | ServerArgs dataclass instantiation; env var overlay | server_args.py | <1% |
| **2. Model loading** | Weight download/read from disk; `model.load_weights()` | model_loader/loader.py, model_runner.py:1371 | 60-80% |
| **3. Memory pool init** | KV cache allocation; `ReqToTokenPool`, `TokenToKVPool` | memory_pool.py, radix_cache.py | 5-10% |
| **4. CUDA graph capture** | Capture decode CUDA graphs for multiple batch sizes | cuda_graph_runner.py:850 | 10-25% |
| **5. Kernel warmup** | FlashInfer autotune, torch.compile warmup, DeepGEMM JIT | model_runner.py:2473 | 5-15% |
| **6. Subprocess launch** | Fork Scheduler + Detokenizer via `mp.Process` | engine.py | <1% |

Evidence: model_runner.py `load_model()`, `initialize()`, `init_device_graphs()`; engine.py `_launch_subprocesses()`

### Known Slow Startup Conditions
- First-time FlashInfer JIT cache population (`disable_flashinfer_autotune` can skip this)
- DeepGEMM JIT compilation on first launch (mitigated by `SGLANG_JIT_DEEPGEMM_PRECOMPILE=true`)
- Large model weight I/O (mitigated by `weight_loader_prefetch_checkpoints`)

---

## 2. Deployment Methods

| Method | Files | Target | Maturity | Risk |
|---|---|---|---|---|
| **Docker (NVIDIA CUDA)** | docker/Dockerfile (830 lines) | Production GPU serving | **High** | Low — multi-stage, multi-CUDA (12.9/13.0), runtime stage for production |
| **Docker (AMD ROCm)** | docker/rocm.Dockerfile (590 lines) | AMD GPU (MI300/MI325/MI35x) | **Medium-High** | Medium — ROCm version dependency |
| **Docker (Ascend NPU)** | docker/npu.Dockerfile (105 lines) | Huawei Ascend (910b/a3) | **Medium** | Medium — CANN version lock |
| **Docker (Intel XPU)** | docker/xpu.Dockerfile (80 lines) | Intel GPU (SYCL/LevelZero) | **Medium** | Medium — Intel ecosystem maturity |
| **Docker (CPU Xeon)** | docker/xeon.Dockerfile (52 lines) | Intel Xeon CPU-only | **Medium** | Low |
| **Docker (ARM64 CPU)** | docker/arm64.Dockerfile (53 lines) | ARM64 CPU-only | **Medium** | Low |
| **Docker (SageMaker)** | docker/sagemaker.Dockerfile (7 lines) + docker/serve (35 lines) | AWS SageMaker hosting | **Medium** | Low — thin wrapper on standard image |
| **Docker (Gateway)** | docker/gateway.Dockerfile (78 lines) | SGLang Model Gateway (router) | **Medium** | Low |
| **Kubernetes (test manifests)** | sgl-model-gateway/e2e_test/k8s_integration/manifests/ | K8s integration testing (kind) | **Low** | High — test-only, not production K8s |
| **PyPI wheel** | pyproject.toml | pip install | **High** | Low |
| **Nightly wheel** | release-pypi-nightly.yml | Pre-release testing | **High** | Medium — unstable API |
| **Source install** | pip install -e "python/" | Development | **High** | Low |

### Docker Image Tags and Registry

| Image | Registry | Tags |
|---|---|---|
| `lmsysorg/sglang` | Docker Hub | `latest`, `v{version}`, `dev`, `dev-cu12`, `nightly-dev-{date}-{sha}` |
| `lmsysorg/sglang` (runtime) | Docker Hub | `latest-runtime`, `v{version}-runtime` |
| `lmsysorg/sglang-rocm` | Docker Hub | `v{version}-rocm700-mi30x`, etc. |
| `rocm/sgl-dev` | Docker Hub | ROCm nightly builds |
| `lmsysorg/sgl-model-gateway` | Docker Hub | `latest`, `v{version}` |

Evidence: .github/workflows/release-docker*.yml

### Deployment Gaps
- **No Helm charts** for production K8s (only test manifests using kind + NodePort)
- **No Terraform/Pulumi/CDK** infrastructure-as-code
- **No systemd unit files** for bare-metal Linux deployments
- **No Kustomize overlays** for environment-specific K8s configuration
- **No HPA (Horizontal Pod Autoscaler)** configurations
- **No service mesh** (Istio/Linkerd) integration examples
- **No official K8s operator** for SGLang

---

## 3. Configuration Reference

### 3.1 Configuration Loading Order

```
1. SGLANG_* environment variables (environ.py Envs descriptors)
   → Loaded at Python import time, before CLI parsing
2. CLI arguments (server_args.py argparse)
   → --model-path, --port, --tp-size, etc. (~200+ flags)
3. ServerArgs post-init (server_args.py _post_init__)
   → Derived values, validation, GPU detection
4. Model config (from HF config.json)
   → Overrides some ServerArgs defaults (context_length, etc.)
5. Runtime env var overrides (via EnvField.override() context manager)
```

Evidence: environ.py:38-106 (EnvField descriptors), server_args.py (argparse + dataclass), model_config.py (HF config parsing)

### 3.2 Critical Configuration Reference

| Key | Type | Default | Required | Secret | Source | Risk |
|---|---|---|---|---|---|---|
| `--model-path` | str | None | **Yes** | No | server_args.py | High — controls which model loads |
| `--host` | str | `127.0.0.1` | No | No | server_args.py | **Dangerous if 0.0.0.0 without auth** |
| `--port` | int | `30000` | No | No | server_args.py | Low |
| `--api-key` | str | None | **Should be** | **Yes** | server_args.py | **High — no auth by default** |
| `--tp-size` | int | `1` | No | No | server_args.py:447 | Medium — affects model parallelism |
| `--dp-size` | int | `1` | No | No | server_args.py:526 | Medium — data parallelism |
| `--ep-size` | int | `1` | No | No | server_args.py | Medium — expert parallelism |
| `--mem-fraction-static` | float | None (auto: ~0.9) | No | No | server_args.py:417 | Medium — OOM if too high |
| `--max-running-requests` | int | None (auto) | No | No | server_args.py:418 | Medium — throughput cap |
| `--max-total-tokens` | int | None (auto) | No | No | server_args.py:420 | Medium — KV cache cap |
| `--max-prefill-tokens` | int | `16384` | No | No | server_args.py:423 | Medium — TTFT vs throughput |
| `--log-level` | str | `info` | No | No | server_args.py:471 | Low — DEBUG may leak PII |
| `--enable-metrics` | bool | False | No | No | server_args.py:482 | Low — must enable for monitoring |
| `--disable-cuda-graph` | bool | False | No | No | server_args.py:708 | **High performance impact** |
| `--disable-radix-cache` | bool | False | No | No | server_args.py:705 | **High performance impact** |
| `--disable-overlap-schedule` | bool | False | No | No | server_args.py:727 | **High performance impact** |
| `--trust-remote-code` | bool | True | No | No | server_args.py | **High security risk** (RCE via model code) |
| `--enable-lora` | bool | False | No | No | server_args.py | Low |
| `--speculative-algorithm` | str | `NONE` | No | No | server_args.py | Medium — affects correctness |
| `--quantization` | str | None | No | No | server_args.py | Medium — affects accuracy |
| `--disaggregation-mode` | str | `null` | No | No | server_args.py | High — multi-node deployment |
| `--stream-interval` | int | `1` | No | No | server_args.py:451 | Low — token streaming granularity |
| `--schedule-policy` | str | `fcfs` | No | No | server_args.py:425 | Medium — starvation risk with lpm |
| `--log-requests` | bool | False | No | No | server_args.py:473 | Medium — may log user prompts |
| `--enable-torch-compile` | bool | False | No | No | server_args.py:735 | Low — experimental speedup |

### 3.3 Key Environment Variables (SGLANG_*)

| Category | Key Env Vars | Count |
|---|---|---|
| **Scheduling** | `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`, `SGLANG_SCHEDULER_MAX_RECV_PER_POLL`, `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_*`, `SGLANG_REQ_WAITING_TIMEOUT`, `SGLANG_REQ_RUNNING_TIMEOUT` | ~20 |
| **Disaggregation** | `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`, `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`, `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`, `SGLANG_DISAGGREGATION_NIXL_BACKEND` | ~15 |
| **Memory** | `SGLANG_EMPTY_CACHE_INTERVAL`, `SGLANG_MEMORY_SAVER_CUDA_GRAPH`, `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` | ~10 |
| **Performance** | `SGLANG_ENABLE_TORCH_COMPILE`, `SGLANG_JIT_DEEPGEMM_PRECOMPILE`, `SGLANG_USE_BREAKABLE_CUDA_GRAPH`, `SGLANG_OPT_FP8_WO_A_GEMM` | ~20 |
| **Observability** | `SGLANG_LOG_MS`, `SGLANG_LOG_SCHEDULER_STATUS_TARGET`, `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS`, `SGLANG_PROFILE_V2` | ~10 |
| **Distributed** | `SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE`, `SGLANG_TCP_STORE_PORT`, `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` | ~5 |
| **Platform-specific** | `SGLANG_USE_AITER` (AMD), `SGLANG_NPU_*` (Ascend), `SGLANG_MUSA_*` (Moore Threads) | ~15 |
| **Plugin** | `SGLANG_PLATFORM`, `SGLANG_PLUGINS` | 2 |

Evidence: environ.py:159-661 (80+ EnvField descriptors documented with inline comments)

### 3.4 Dangerous Defaults

| Default | Risk | Recommendation |
|---|---|---|
| `--host 127.0.0.1` — safe default | Low | Keep as-is |
| `--api-key None` — no auth | **High** — open access to all endpoints | Always set `--api-key` in non-localhost deployments |
| `--trust-remote-code True` — arbitrary code exec | **High** — model code can execute arbitrary Python | Set `--disable-trust-remote-code` for untrusted models |
| `--log-level info` — may leak prompt data at DEBUG | Medium | Set to `warning` in production |
| `--enable-metrics False` — no monitoring | Medium | Enable for production |
| `--host 0.0.0.0` (user choice) — exposes to network | **High** if no auth | Require API key when binding to all interfaces |

---

## 4. CI/CD Pipeline

### 4.1 Overall CI Architecture

**Total workflows:** 74 GitHub Actions `.yml` files
**No non-GitHub CI found** (no Jenkinsfile, .gitlab-ci.yml, etc.)

Evidence: .github/workflows/ directory (74 files)

### 4.2 PR Testing Pipeline (`pr-test.yml`)

| Stage | Hardware | Tests | Timeout |
|---|---|---|---|
| **Check changes** | ubuntu-latest | Detect changed components; compute partitions | — |
| **Gate** | ubuntu-latest | Draft PR block, `run-ci` label requirement, rate limit | — |
| **sgl-kernel build** | x64/arm kernel build nodes | Build CUDA 12.9 + 13.0 wheels | — |
| **Stage A** | 1×5090, ubuntu-latest (CPU) | Small 1-GPU tests + CPU tests | 10 min |
| **Stage B** | 1×5090, 1×H100, 2×H100, 4×B200 | Medium/large GPU tests | 30 min |
| **Stage C** | 4×H100, 8×H200, 8×H20, 4×B200 | Multi-GPU, DeepEP, DSv4, RDMA tests | 30-60 min |
| **Multimodal gen** | Varied GPU | Multimodal/diffusion tests | — |
| **Finish** | ubuntu-latest | Aggregate all results | — |

Evidence: .github/workflows/pr-test.yml (642 lines)

### 4.3 Platform-Specific PR Tests

| Platform | Workflow | Runner |
|---|---|---|
| **NVIDIA (main)** | pr-test.yml | 5090, H100, H200, H20, B200, GB200 |
| **AMD ROCm** | pr-test-amd.yml | MI300, MI325, MI35x |
| **ARM64** | pr-test-arm64.yml | ARM runners |
| **Intel XPU** | pr-test-xpu.yml | XPU runners |
| **Intel Xeon (CPU)** | pr-test-xeon.yml | ubuntu-latest |
| **Huawei Ascend NPU** | pr-test-npu.yml | Ascend aarch64 |
| **Moore Threads MUSA** | pr-test-musa.yml | MUSA runners |
| **Rust gRPC** | pr-test-rust.yml | ubuntu-latest |

Evidence: .github/workflows/pr-test-*.yml

### 4.4 Release Pipeline

```
[bot-bump version] → [release-branch-cut.yml] → [release-tag.yml]
                                                        ↓
                    ┌───────────────────────────────────┼───────────────────────────────┐
                    ↓                                   ↓                               ↓
    [release-pypi.yml]                    [release-docker.yml]              [release-docs.yml]
    Python 3.10-3.13 wheels               docker: framework_final            Sphinx HTML →
    x86_64 + aarch64                      tags: v{ver}, latest               sgl-project.github.io
                                          [release-docker-runtime.yml]
                                          docker: runtime stage
                                          [release-docker-amd.yml]
                                          docker: ROCm
                                          [release-docker-npu.yml]
                                          docker: Ascend NPU
```

Evidence: .github/workflows/release-*.yml

### 4.5 Nightly CI

| Workflow | Schedule | Coverage |
|---|---|---|
| **nightly-test-nvidia.yml** | Daily 00 UTC | 1/2/4/8-GPU NVIDIA tests, accuracy + performance, diffusion |
| **nightly-test-amd.yml** | Daily 17:30 UTC | 1/2/4/8-GPU AMD tests, DeepSeek-V3.1/V3.2, Grok, Qwen3, etc. |
| **nightly-72-gpu-gb200.yml** | Daily 02 UTC | Slurm-based 72-GPU GB200 benchmarks |
| **release-docker-dev.yml** | Daily 00 UTC | Nightly Docker dev images |
| **release-pypi-nightly.yml** | Daily 02 UTC | Nightly PyPI wheel |
| **trivy-scan-dev.yml** | Daily 06 UTC | Trivy vulnerability scan on Docker images |
| **ci-failure-monitor.yml** | Every 12h | CI failure root cause analysis |
| **weekly-test-nvidia.yml** | Weekly Sunday | 8-GPU H200 weekly tests |

Evidence: .github/workflows/nightly-*.yml, weekly-*.yml

### 4.6 Quality Gates

| Gate | Tool | Enforcement |
|---|---|---|
| **Pre-commit** | ruff, isort, black, codespell, clang-format | Local + CI (lint.yml) |
| **PR gate** | pr-gate.yml | Blocks draft PRs, requires `run-ci` label, rate-limits low-permission users |
| **Docker security scan** | Trivy | Daily SARIF upload to GitHub Security tab |
| **Test partitioning** | compute_partitions.py + sglang-ci-stats | Dynamic, ML-driven partition sizing |
| **CI failure analysis** | ci-failure-monitor.yml | Automated root cause analysis |
| **Auto-bisect** | ci-auto-bisect.yml | Manual trigger for regression hunting |

### 4.7 CI Automation Bots

| Bot | Purpose |
|---|---|
| **bot-bump-flashinfer-version.yml** | Auto-bump flashinfer dependency |
| **bot-bump-kernel-version.yml** | Auto-bump sgl-kernel version |
| **bot-bump-sglang-version.yml** | Auto-bump SGLang version |
| **labeler.yml** | Auto-label PRs based on changed files |
| **close-inactive-issues.yml** | Auto-close issues with 60+ days inactivity |
| **cancel-pr-workflow-on-merge.yml** | Cancel running PR workflows on merge |

Evidence: .github/workflows/bot-*.yml

### 4.8 CI Maturity Assessment: A (Production-Grade)

**Strengths:**
- 8 hardware platforms tested with multi-GPU configurations
- Smart change detection avoids wasted CI (only test what changed)
- ML-driven dynamic test partitioning for optimal job sizing
- PR gate with rate limiting for resource protection
- Comprehensive multi-arch Docker publishing (x86_64 + aarch64, CUDA 12/13, ROCm, NPU)
- Security scanning integrated (Trivy → GitHub Security)
- Automated dependency bumping reduces maintenance burden

**Gaps:**
- No code coverage integration for GPU test suites (only unit test coverage)
- No SAST/DAST beyond container scanning
- No SBOM generation in release pipeline
- No automated performance regression alerting (metrics collected but not blockingly enforced)

---

## 5. Observability

### 5.1 Observability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SGLang Server                             │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐       │
│  │ TokenizerMgr │  │  Scheduler   │  │ DetokenizerMgr   │       │
│  │              │  │              │  │                  │       │
│  │ metrics ─────┼──┼─ metrics ────┼──┼─ (via IPC)       │       │
│  │ trace ───────┼──┼─ trace ──────┼──┼─ (via IPC)       │       │
│  └──────────────┘  └──────┬───────┘  └──────────────────┘       │
│                           │                                       │
│              ┌────────────┼────────────┐                         │
│              ▼            ▼            ▼                          │
│        /metrics      /health      /v1/loads                      │
│        (Prometheus)  (liveness)   (detailed)                     │
│              │                                        │           │
│              ▼                                        ▼           │
│     Prometheus scrape                         OTLP gRPC/HTTP     │
│     (multiprocess mode)                       (to Jaeger/OTel)   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Logging

| Aspect | Configuration | Default | Evidence |
|---|---|---|---|
| **Log level** | `--log-level` | `info` | server_args.py:471 |
| **HTTP log level** | `--log-level-http` | Same as `--log-level` | server_args.py:472 |
| **Request logging** | `--log-requests` (bool) | False | server_args.py:473 |
| **Request log level** | `--log-requests-level` (0-3) | 3 | server_args.py:474 |
| **Request log format** | `--log-requests-format` | `text` (also `json`) | server_args.py:475 |
| **Request log target** | `--log-requests-target` | stdout | server_args.py:476 |
| **File rotation** | `TimedRotatingFileHandler` | Hourly | log_utils.py:40 |
| **Scheduler status** | `SGLANG_LOG_SCHEDULER_STATUS_TARGET` | Disabled | environ.py:174 |
| **GC logging** | `SGLANG_LOG_GC` | False | environ.py:169 |
| **Millisecond precision** | `SGLANG_LOG_MS` | False | environ.py:171 |
| **Request exceeded warning** | `SGLANG_LOG_REQUEST_EXCEEDED_MS` | -1 (off) | environ.py:172 |

**Log format:** `[YYYY-MM-DD HH:MM:SS] message`
Evidence: server_args.py:7738-7741

### 5.3 Metrics (Prometheus)

**Enablement:** `--enable-metrics` flag (server_args.py:482)
**Endpoint:** `GET /metrics`

#### Core Request Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `sglang:num_running_reqs` | Gauge | priority | Running request count |
| `sglang:num_queue_reqs` | Gauge | — | Queued request count |
| `sglang:gen_throughput` | Gauge | — | Token generation throughput (tokens/s) |
| `sglang:cache_hit_rate` | Gauge | — | Prefix cache hit rate |
| `sglang:time_to_first_token_seconds` | Histogram | — | TTFT distribution |
| `sglang:inter_token_latency_seconds` | Histogram | — | ITL distribution |
| `sglang:e2e_request_latency_seconds` | Histogram | — | End-to-end latency |
| `sglang:queue_time_seconds` | Histogram | — | Request queue waiting time |
| `sglang:num_requests_total` | Counter | — | Total requests |
| `sglang:prompt_tokens_total` | Counter | — | Total prompt tokens processed |
| `sglang:generation_tokens_total` | Counter | — | Total generated tokens |
| `sglang:num_aborted_requests_total` | Counter | — | Aborted request count |

#### Memory Pool Metrics

| Metric | Type | Description |
|---|---|---|
| `sglang:token_usage` | Gauge | KV cache utilization ratio (max of full/SWA/mamba) |
| `sglang:full_token_usage` | Gauge | Full-attention KV pool usage |
| `sglang:swa_token_usage` | Gauge | SWA pool usage |
| `sglang:num_used_tokens` | Gauge | Absolute used token count |
| `sglang:kv_available_tokens` | Gauge | Available KV cache tokens |
| `sglang:kv_evictable_tokens` | Gauge | Evictable cache tokens |

#### Speculative Decoding Metrics

| Metric | Type | Description |
|---|---|---|
| `sglang:spec_accept_length` | Gauge | Mean accepted tokens per verify step |
| `sglang:spec_accept_rate` | Gauge | Acceptance rate |

#### Disaggregation Metrics

| Metric | Type | Description |
|---|---|---|
| `sglang:kv_transfer_speed_gb_s` | Histogram | KV transfer throughput |
| `sglang:kv_transfer_latency_ms` | Histogram | KV transfer latency |
| `sglang:num_prefill_bootstrap_queue_reqs` | Gauge | Prefill bootstrap queue depth |
| `sglang:num_prefill_inflight_queue_reqs` | Gauge | Inflight prefill queue depth |
| `sglang:num_decode_prealloc_queue_reqs` | Gauge | Decode prealloc queue depth |

#### Execution Performance Metrics

| Metric | Type | Description |
|---|---|---|
| `sglang:realtime_tokens_total` | Counter | Tokens processed by mode (prefill_compute, prefill_cache, decode) |
| `sglang:forward_execution_seconds_total` | Counter | GPU busy time by category |
| `sglang:fwd_occupancy` | Gauge | GPU occupancy percentage |
| `sglang:utilization` | Gauge | Request-based utilization |
| `sglang:estimated_flops_per_gpu_total` | Counter | Estimated FLOPs (for MFU calculation) |

Evidence: metrics_collector.py (1779 lines), ~50+ metric families defined

#### Additional Metrics Collectors

| Collector | Location | Purpose |
|---|---|---|
| **SchedulerMetricsCollector** | metrics_collector.py | Core scheduler stats |
| **TokenizerMetricsCollector** | metrics_collector.py:1255 | Tokenizer throughput, latency |
| **StorageMetricsCollector** | metrics_collector.py:1563 | HiCache stats |
| **CPU Monitor** | cpu_monitor.py | Per-component CPU time |
| **Function Timer** | func_timer.py | Decorated function latency |

### 5.4 Distributed Tracing (OpenTelemetry)

| Aspect | Configuration | Evidence |
|---|---|---|
| **Protocol** | OTLP over gRPC or HTTP/protobuf | trace.py:161 |
| **Exporter** | `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (default: grpc) | trace.py:39-60 |
| **Schedule delay** | `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS` (default: 500ms) | environ.py:209 |
| **Max batch size** | `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE` (default: 64) | environ.py:210 |
| **Trace level** | `SGLANG_TRACE_LEVEL` (0-5, default: 3), runtime adjustable via `/set_trace_level` | trace.py:35, http_server.py:983 |
| **W3C propagation** | `traceparent`, `tracestate` headers | trace.py:37 |
| **Gen-AI semantics** | `gen_ai.usage.*`, `gen_ai.request.*`, `gen_ai.response.*`, `gen_ai.latency.*` | trace.py:698 |
| **Cross-process** | Trace context serialized across TokenizerManager → Scheduler → Detokenizer via IPC | trace.py:249 `TraceReqContext` |
| **Dependencies** | `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc` | pyproject.toml:120-124 |

### 5.5 Health Checks

| Endpoint | Method | Checks | Timeout | Evidence |
|---|---|---|---|---|
| `/health` | GET | Server status (starting/waiting/ready), graceful shutdown flag | — (instant if `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`) | http_server.py:505-577 |
| `/health_generate` | GET | Same + runs actual 1-token inference | `SGLANG_HEALTH_CHECK_TIMEOUT` (default: 20s) | http_server.py:505-577 |
| `/v1/loads` | GET | Comprehensive per-DP-rank metrics (running/waiting reqs, token usage, throughput, utilization) | — | v1_loads.py |
| `/get_server_info` | GET | Server config info | — | http_server.py |
| Non-rank-0 nodes | — | Dummy `/health` + `/health_generate` (always 200) | — | common.py:2579 |

### 5.6 Profiling

| Tool | Activation | Output | Evidence |
|---|---|---|---|
| **PyTorch Profiler** | `POST /start_profile` + `POST /stop_profile` | Chrome trace (`.trace.json.gz`) | http_server.py:947-980 |
| **ROCm RPD** | Same endpoint (auto-detected on AMD) | RPD trace | scheduler_profiler_mixin.py |
| **CUDA Memory** | `--enable-metrics` + profile | Memory trace | scheduler_profiler_mixin.py |
| **Profile V2** | `SGLANG_PROFILE_V2=true` | Stage-based (prefill vs decode) traces | profile_utils.py |
| **Client profiler** | `python -m sglang.profiler` | Downloads trace from server | profiler.py |
| **Profile merger** | Built-in | Merges multi-rank traces | profile_merger.py |
| **Expert distribution** | `/start_expert_distribution_record` | Expert routing heatmap | http_server.py:1006-1036 |

### 5.7 Pre-built Dashboards

| Asset | Path | Content |
|---|---|---|
| **Grafana dashboard** | examples/monitoring/grafana/dashboards/json/sglang-dashboard.json | E2E latency, TTFT, running reqs, throughput, cache hit rate, queued reqs |
| **Prometheus config** | examples/monitoring/prometheus.yaml | 5s scrape interval, targets localhost:30000 |
| **Docker Compose** | examples/monitoring/docker-compose.yaml | Prometheus + Grafana (host network) |
| **Tracing stack** | examples/monitoring/tracing_compose.yaml | OTel Collector + Jaeger |

### 5.8 Observability Gaps

| Gap | Impact | Recommendation |
|---|---|---|
| **No built-in alerting rules** | Manual threshold monitoring | Add Prometheus alert rules for latency, OOM, queue depth |
| **No SLO tracking** | No latency/availability targets defined | Define SLOs for TTFT, throughput, availability |
| **No audit logging** | Security events not tracked | Add request-level audit log with timestamps, IPs, auth |
| **No log aggregation** | Multi-node logs not centralized | Document ELK/Loki integration |
| **No distributed tracing for multi-node** | PD disaggregation + DP cross-node tracing incomplete | Extend trace context propagation for cross-node KV transfer |
| **Metrics disabled by default** | No monitoring without `--enable-metrics` | Enable lightweight metrics always |

---

## 6. Health Check And Runtime Operations

### 6.1 Health Endpoint Details

**`/health` (lightweight):**
- Returns 200 if server is ready (not in `Starting` state, not `gracefully_exit`)
- Returns 503 if starting or shutting down
- Optionally runs 1-token inference if `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=true` (default)
- Special request ID prefix (`HEALTH_CHECK_RID_PREFIX`) to exclude from metrics

**`/health_generate` (deep):**
- Same as `/health` but always runs real inference (cannot be disabled)
- 20-second default timeout
- Ensures the full pipeline (TokenizerManager → Scheduler → ModelRunner → Detokenizer) is functional

Evidence: http_server.py:505-577, constants.py:12

### 6.2 Runtime Configuration Changes

| Operation | Method | Notes |
|---|---|---|
| **Change log level** | `POST /set_log_level` | Not found — done via `--log-level` at startup only |
| **Change trace level** | `POST /set_trace_level` | Dynamic OpenTelemetry trace verbosity |
| **Hot-reload weights** | `POST /update_weights_from_disk` or `POST /update_weights_from_tensor` | Requires auth if `--api-key` set |
| **Flush KV cache** | `POST /flush_cache` | Clears all cached prefixes |
| **Freeze GC** | `POST /freeze_gc` | For profiling scenarios |

### 6.3 Graceful Shutdown

| Signal | Behavior | Evidence |
|---|---|---|
| **SIGTERM / SIGINT** | Sets `gracefully_exit=True` in _GlobalState; `/health` returns 503; waits for in-flight requests to complete | http_server.py _GlobalState |
| **Force kill** | `killall_sglang` CLI command kills all SGLang processes | cli/killall.py |
| **ZMQ cleanup** | Scheduler closes ZMQ sockets; subprocesses joined with timeout | engine.py shutdown |

### 6.4 Known Runtime Failure Modes

| Failure | Detection | Recovery | Evidence |
|---|---|---|---|
| **Scheduler crash** | Watchdog (watchdog.py) detects subprocess exit | Engine restarts scheduler subprocess | watchdog.py |
| **CUDA OOM** | PyTorch exception during forward | Request failed; scheduler evicts cache and retries | scheduler.py OOM handling |
| **Grammar compilation timeout** | Future timeout in grammar_queue | Returns `InvalidGrammarObject` | grammar_manager.py |
| **ZMQ connection lost** | ZMQ exception in recv loop | Subprocess detected via watchdog, restarted | watchdog.py |
| **Slow rank detection** | `SGLANG_DETECT_SLOW_RANK` env var | Logs slow NCCL operations | environ.py:190 |

---

## 7. Scaling And Reliability

### 7.1 Scaling Dimensions

| Dimension | Mechanism | Range | Key Config | Evidence |
|---|---|---|---|---|
| **Tensor Parallelism (TP)** | Megatron-style layer sharding | 1-8 GPUs (single node) | `--tp-size` | server_args.py:447 |
| **Pipeline Parallelism (PP)** | 1F1B schedule | 2-N GPUs | `--pp-size` | server_args.py |
| **Expert Parallelism (EP)** | DeepEP/FlashInfer/Standard dispatchers | 1-N GPUs | `--ep-size` | server_args.py |
| **Data Parallelism (DP)** | Data-parallel attention | 2-N GPUs | `--dp-size`, `--enable-dp-attention` | server_args.py:526 |
| **PD Disaggregation** | Separate prefill + decode servers | 2+ servers | `--disaggregation-mode prefill/decode` | disaggregation/ |
| **Attention DP** | Shard attention across DP ranks | 2-N GPUs | `--enable-dp-attention` | server_args.py:729 |
| **Multi-node** | Cross-machine communication | N nodes × M GPUs | `--nnodes`, SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE | environ.py:285 |

### 7.2 Concurrency And Throttling

| Mechanism | Default | Config | Evidence |
|---|---|---|---|
| **Max running requests** | Auto (based on memory) | `--max-running-requests` | server_args.py:418 |
| **Max queued requests** | Auto | `--max-queued-requests` | server_args.py:419 |
| **Max total tokens** | Auto | `--max-total-tokens` | server_args.py:420 |
| **Max prefill tokens per batch** | 16384 | `--max-prefill-tokens` | server_args.py:423 |
| **Request waiting timeout** | -1 (disabled) | `SGLANG_REQ_WAITING_TIMEOUT` (seconds) | environ.py:264 |
| **Request running timeout** | -1 (disabled) | `SGLANG_REQ_RUNNING_TIMEOUT` (seconds) | environ.py:266 |
| **Rate limiting** | **None built-in** | N/A | Gap |
| **Token budgets per user** | **None built-in** | N/A | Gap |
| **Queue priority** | `--schedule-policy` (fcfs/lpm/dfs-weight/lof) | `--schedule-policy` | server_args.py:425 |

### 7.3 KV Cache Scaling

| Aspect | Mechanism | Config | Evidence |
|---|---|---|---|
| **Prefix sharing** | RadixCache (trie-based) | `--disable-radix-cache` to disable | radix_cache.py |
| **Eviction policies** | LRU, LFU, FIFO, MRU, FILO, SLRU, Priority | `--cache-eviction-policy` | evict_policy.py |
| **Hierarchical cache** | GPU → CPU → Storage tiering | `--enable-hierarchical-cache` | HiCache |
| **Page-based allocation** | Reduces external fragmentation | `--page-size` | allocator.py |
| **SWA/KV split** | Separate pools for full-attn, SWA, Mamba | `--enable-unified-radix-tree` | memory_pool.py |

### 7.4 Stateless vs Stateful

SGLang is **stateful** — KV cache state persists across requests for prefix sharing:
- **Shared state:** RadixCache (prefix tree), memory pools (KV tensors)
- **Isolation:** LoRA adapters are per-request; MPS (CUDA Multi-Process Service) for GPU isolation
- **Stateless components:** TokenizerManager (request tracking only), HTTP server

### 7.5 Reliability Features

| Feature | Status | Evidence |
|---|---|---|
| **Watchdog (subprocess monitor)** | Built-in | watchdog.py |
| **CUDA graph fallback** | Falls back to eager execution if graph can't run | cuda_graph_runner.py `can_run()` |
| **OOM recovery** | Retract requests, evict cache, re-run | scheduler.py retraction logic |
| **Grammar timeout** | ThreadPool future timeout | grammar_manager.py |
| **Disaggregation heartbeat** | 5s interval, 2 failures → disconnect | environ.py:240-241 |
| **Stuck scheduler detection** | Test-only (`SGLANG_TEST_STUCK_SCHEDULER_INIT`) | environ.py:193 |
| **Automatic cache purge** | `SGLANG_EMPTY_CACHE_INTERVAL` for memory accumulation | environ.py:253 |

---

## 8. Backup / Migration / Rollback

### 8.1 Model Weights

| Aspect | Mechanism | Evidence |
|---|---|---|
| **Backup** | None built-in — model weights from HF Hub/S3/local; rely on source of truth | model_loader/loader.py |
| **Versioning** | Model weights identified by model_path + revision; no internal versioning | model_loader/ |
| **Hot-reload** | `POST /update_weights_from_disk` or `POST /update_weights_from_tensor` | http_server.py |
| **CPU backup** | `--enable-weights-cpu-backup` copies weights to CPU RAM | server_args.py |

### 8.2 KV Cache

| Aspect | Mechanism | Evidence |
|---|---|---|
| **Persistence** | HiCache supports Redis/S3/file backends for KV cache backup | hicache/ |
| **Restore** | HiCache can load backed-up KV pages on cold start | hicache/ |
| **Disaggregation backup** | `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` for KV transfer retry | environ.py:239 |

### 8.3 Server Configuration

| Aspect | Status |
|---|---|
| **Config versioning** | No explicit version — server_args.py changes tracked by git |
| **Rollback** | `pip install sglang==<previous_version>` + restart |
| **Schema migration** | Not applicable (no persistent DB) |

### 8.4 What's Missing

| Gap | Impact | Recommendation |
|---|---|---|
| **No model artifact management** | Can't track which model version served when | Use MLflow/Weights & Biases for model registry |
| **No KV cache snapshot** | Pre-warming requires fresh requests after restart | Use HiCache storage backends for warm restarts |
| **No canary deployment** | Can't gradually roll out new config/models | Use API gateway for traffic splitting |
| **No blue-green deployment** | No formal zero-downtime deploy strategy | Implement via K8s rolling update or 2-pool setup |
| **No A/B testing infrastructure** | Can't compare model versions with live traffic | Use model gateway with traffic splitting |

---

## 9. Performance Bottleneck Hypotheses

### 9.1 Hot Path Analysis

| Hot Path | Location | Frequency | Bottleneck Type | Evidence |
|---|---|---|---|---|
| **Scheduler event loop** | scheduler.py `event_loop_normal/overlap` | Every batch (~10-100ms) | CPU — ZMQ recv, batch formation | scheduler.py:1537,1564 |
| **KV cache lookup** | radix_cache.py `match_prefix()` | Every request enqueue | CPU — trie traversal O(depth) | radix_cache.py:360 |
| **Attention forward** | flashinfer_backend.py `forward_decode/forward_extend` | Every batch | GPU — memory bandwidth bound | model_runner.py forward() |
| **CUDA graph replay** | cuda_graph_runner.py `replay()` | Every decode batch | GPU — kernel launch overhead | cuda_graph_runner.py:1307 |
| **ZMQ serialization** | scheduler.py `recv_requests()` | Every poll loop (~1ms) | CPU — pickle deserialization | scheduler.py:1656 |
| **Token sampling** | sampler.py `Sampler.forward()` | Every batch | CPU/GPU — top-k/top-p, grammar accept | sampler.py |
| **Detokenization** | detokenizer_manager.py | Every step (~50 tokens) | CPU — HF tokenizer decode | detokenizer_manager.py |
| **RadixCache eviction** | radix_cache.py `evict()` | On memory pressure | CPU — heap operations O(N log N) | radix_cache.py:560 |
| **KV cache allocation** | allocator.py `alloc()` | Every extend batch | CPU — page search and merge | allocator.py:148 |

### 9.2 Known Performance Tuning Parameters

| Parameter | Default | Tuning Guidance | Impact |
|---|---|---|---|
| `--mem-fraction-static` | Auto (~0.9) | Lower if OOM, higher for more KV cache | KV cache capacity |
| `--max-prefill-tokens` | 16384 | Higher for throughput, lower for TTFT | Prefill latency vs batch efficiency |
| `--cuda-graph-max-bs` | Auto | Limits CUDA graph capture range | Decode latency |
| `--schedule-conservativeness` | 1.0 | <1.0 for aggressive, >1.0 for conservative | Token budget estimation |
| `--page-size` | Auto | Smaller for less fragmentation, larger for efficiency | Memory utilization |
| `--disable-overlap-schedule` | False | Never disable in production | CPU/GPU parallelism |
| `--disable-cuda-graph` | False | Never disable in production | Decode latency (2-5x slower) |
| `--disable-radix-cache` | False | Disable only if no prefix reuse | Throughput |
| `--stream-interval` | 1 | Higher for less IPC, lower for smoother streaming | Client-perceived latency |
| `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` | -1 (unlimited) | Limit if many concurrent requests cause starvation | Fairness vs throughput |
| `SGLANG_EMPTY_CACHE_INTERVAL` | -1 (disabled) | Set if memory accumulates over long serving | Memory leak mitigation |

### 9.3 Performance Hypotheses (Not Verified — No GPU Available)

These are static-analysis-based hypotheses that require dynamic profiling to validate:

| # | Hypothesis | Evidence (Static) | Validation Method | Priority |
|---|---|---|---|---|
| H1 | RadixCache `inc_lock_ref`/`dec_lock_ref` traversal is O(seq_len) and may become bottleneck under long-context workloads | radix_cache.py:589-623 walks parent chain to root | Profile with 128K+ context; benchmark lock_ref latency | Medium |
| H2 | ZMQ `recv_pyobj(NOBLOCK)` in polling loop is CPU-intensive under high request rate | scheduler.py:1656, single-threaded event loop | Profile CPU usage under 1000+ req/s; measure recv loop time | High |
| H3 | `tolist()` calls on output tokens take significant CPU for large batch sizes (>256) | scheduler_output_processor_mixin.py:211-232, Python list conversion | Profile `process_batch_result` by batch size | Medium |
| H4 | `_update_prefill_budget` page overhead calculation is too conservative, wasting ~5-10% KV cache | schedule_policy.py:584-593, comment "may be too conservative" | Measure actual vs theoretical KV cache utilization | Low |
| H5 | CUDA graph `bisect_left` + buffer population overhead grows with `cuda_graph_max_bs` | cuda_graph_runner.py:1244-1247, linear sweep | Profile replay_prepare() time vs CUDAGraph max_bs | Low |
| H6 | DeepGEMM JIT compilation on first launch can take 5-15 minutes (blocking startup) | model_runner.py:2507, `SGLANG_JIT_DEEPGEMM_PRECOMPILE` exists as mitigation | Measure startup time with/without precompiled cache | Medium |
| H7 | Overlap disabled for spec_v2 + grammar + decode combinations, reducing throughput | scheduler.py:1639-1648 TODO comment | Benchmark spec+grammar throughput with/without overlap | Medium |
| H8 | `SGLANG_FORCE_STREAM_INTERVAL=50` masks true TTFT for non-streaming requests | environ.py:274, comment explains TTFT accuracy trade-off | Compare TTFT at stream_interval=1 vs 50 | Low |

### 9.4 Model Forward Performance Characteristics

| Model Size | Attention | Bottleneck | Evidence |
|---|---|---|---|
| **Small (<7B)** | MHA | Kernel launch overhead, CPU/GPU sync | CUDA graph mitigation |
| **Medium (7B-70B)** | MHA/GQA | Memory bandwidth (KV cache reads) | FlashInfer paged attention |
| **Large (70B+)** | GQA/MQA | Compute bound (MLP), communication (TP all-reduce) | Tensor parallelism required |
| **MoE (DeepSeek-V3)** | MLA | Expert dispatch (all-to-all), EP communication | DeepEP, Mega MoE mitigations |
| **Ultra-large (DSv4)** | DSA + MLA | Compressed attention overhead, multi-dimensional parallelism | Custom DSv4 attention backends |

---

## 10. Load Test Suggestions

### 10.1 Benchmarking Tools (Built-in)

| Tool | Path | Purpose |
|---|---|---|
| **bench_one_batch** | test/bench_one_batch_server_internal.py | Single-batch microbenchmark (TPOT, ITL, TTFT) |
| **NightlyBenchmarkRunner** | test/performance_test_runner.py | Parameterized multi-config benchmark |
| **Nightly accuracy + perf** | test/accuracy_test_runner.py | Combined accuracy and performance |
| **stress-test.yml** | .github/workflows/stress-test.yml | CI stress test (configurable prompts + duration) |

### 10.2 Recommended Load Test Scenarios

#### Level 1: Basic (Should Pass Before Any Deployment)

```bash
# 1. Single request latency
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 50}}'

# 2. Concurrent requests baseline
for i in {1..10}; do
  curl -X POST http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 100}}' &
done; wait

# 3. Health check under load
watch -n 1 'curl -s http://localhost:30000/health'
```

#### Level 2: Throughput Saturation

- **Tool:** `locust` or custom script using `aiohttp`
- **Target:** Find max throughput (tokens/s) at P99 TTFT < 1s
- **Vary:** Input length (128, 512, 2048, 8192), output length (64, 256, 1024)
- **Metrics:** Throughput, TTFT, ITL, E2E latency, cache hit rate, GPU utilization

#### Level 3: Stress And Reliability

- **Sustained load:** 80% max throughput for 24 hours — monitor memory leak, degradation
- **Burst test:** 10x normal load for 5 minutes — verify queue behavior, no OOM
- **Large request test:** 128K context length — verify memory allocation, eviction
- **Concurrent connections:** 1000+ simultaneous — verify connection handling
- **Abort storm:** Rapid request creation and cancellation — verify cleanup

#### Level 4: Multi-node

- **PD disaggregation:** Measure KV transfer latency, bootstrap time
- **TP scaling efficiency:** Compare 1/2/4/8-GPU throughput (target >85% scaling at 8-GPU)
- **DP scaling:** Compare 1/2/4-DP-rank throughput
- **EP scaling:** Expert load balance under skewed token routing

### 10.3 Metrics To Monitor During Load Tests

| Metric | Acceptable Range | Alert Threshold |
|---|---|---|
| **P50 TTFT** | <100ms for short prompts | >500ms |
| **P99 TTFT** | <1s | >5s |
| **P50 ITL** | <20ms | >50ms |
| **P99 ITL** | <100ms | >500ms |
| **Throughput** | — (baseline-dependent) | <50% of baseline |
| **Cache hit rate** | >50% for repeated prefixes | <20% |
| **KV cache utilization** | 60-85% | >95% (OOM risk) or <30% (over-provisioned) |
| **Queue depth** | <10 | >100 |
| **GPU utilization** | >80% | <50% (indicating CPU bottleneck) |
| **Error rate** | <0.1% | >1% |
| **Request timeout rate** | <0.01% | >0.1% |

---

## 11. Production Deployment Checklist

### 11.1 Pre-Deployment

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Run vulnerability scan on dependencies | ❌ Not done | `pip-audit` or `safety check` |
| 2 | Scan Docker image | ⚠️ CI only (Trivy) | Run locally with `trivy image lmsysorg/sglang:latest` |
| 3 | Review model license for commercial use | ⚠️ Manual | Model license ≠ SGLang license |
| 4 | Test model accuracy on representative dataset | ❌ Not done | Compare logprobs/loss with reference |
| 5 | Benchmark performance baseline | ❌ Not done | TTFT, ITL, throughput at expected load |
| 6 | Verify model loading time acceptable for restart SLA | ❌ Not done | — |

### 11.2 Security Hardening

| # | Item | Status | Configuration |
|---|---|---|---|
| 7 | **Set API key** | ❌ **REQUIRED** | `--api-key <strong-random-key>` |
| 8 | **Bind to localhost** (if behind reverse proxy) | ✅ Default | `--host 127.0.0.1` |
| 9 | **Enable TLS** | ⚠️ Via reverse proxy | nginx/Caddy with Let's Encrypt |
| 10 | **Set production log level** | ⚠️ | `--log-level warning` |
| 11 | **Disable trust_remote_code** (if untrusted model) | ⚠️ | `--disable-trust-remote-code` |
| 12 | Configure firewall (only expose proxy, not SGLang directly) | ❌ Manual | iptables/security groups |
| 13 | Run container as non-root | ❌ Not default | Docker USER directive or K8s securityContext |
| 14 | Set resource limits (CPU/memory) if using orchestrator | ❌ Manual | K8s resource limits/requests |

### 11.3 Observability

| # | Item | Status | Configuration |
|---|---|---|---|
| 15 | **Enable Prometheus metrics** | ⚠️ | `--enable-metrics` |
| 16 | Deploy Prometheus + Grafana | ⚠️ Manual | Use examples/monitoring/ as starting point |
| 17 | Configure alerting rules (latency, OOM, error rate) | ❌ Not provided | Custom Prometheus alert rules |
| 18 | Enable request logging if audit trail needed | ⚠️ | `--log-requests --log-requests-format json --log-requests-target /var/log/sglang/` |
| 19 | Set up OTEL tracing (if multi-service) | ⚠️ | `pip install sglang[tracing]` + OTEL endpoint |
| 20 | Configure health check probes | ⚠️ | K8s: `livenessProbe: /health`, `readinessProbe: /health_generate` |

### 11.4 Reliability

| # | Item | Status | Configuration |
|---|---|---|---|
| 21 | Set request timeout | ⚠️ | `SGLANG_REQ_RUNNING_TIMEOUT=<seconds>` |
| 22 | Set max running requests | ⚠️ | `--max-running-requests <N>` |
| 23 | Set max queued requests | ⚠️ | `--max-queued-requests <N>` |
| 24 | Set up automatic restart (systemd/Docker restart policy/K8s) | ❌ Manual | `Restart=always` or K8s `restartPolicy: Always` |
| 25 | Plan for graceful shutdown (drain in-flight requests) | ⚠️ | SGLang supports SIGTERM graceful exit |
| 26 | Test OOM recovery under load | ❌ Not tested | — |
| 27 | Plan for model update without downtime | ❌ Not automated | Blue-green or rolling update strategy |

### 11.5 Scaling

| # | Item | Status | Configuration |
|---|---|---|---|
| 28 | Right-size `--mem-fraction-static` | ⚠️ | Balance weight memory vs KV cache |
| 29 | Choose TP/PP size based on model + GPU count | ⚠️ | Benchmark different parallelism configs |
| 30 | Consider PD disaggregation for >10B models | ⚠️ | `--disaggregation-mode prefill/decode` |
| 31 | Set up API gateway / load balancer | ❌ Manual | nginx/HAProxy/K8s Ingress + SGLang Gateway |
| 32 | Configure auto-scaling (K8s HPA or custom) | ❌ Manual | Based on `sglang:num_queue_reqs` metric |

### 11.6 Backup And Recovery

| # | Item | Status | Notes |
|---|---|---|---|
| 33 | Document model source (HF Hub revision/S3 path) | ❌ Manual | For reproducible deployment |
| 34 | Plan for KV cache warm-up after restart | ❌ | HiCache or replay production traces |
| 35 | Test full restart cycle (stop → start → healthy) | ❌ Not tested | — |

### 11.7 Operations Runbook

| Scenario | Action |
|---|---|
| **Server won't start** | Run `python -m sglang.check_env` → check CUDA, GPU memory, port availability |
| **OOM during serving** | Reduce `--max-running-requests`, `--mem-fraction-static`, or `--cuda-graph-max-bs` |
| **High latency** | Check Prometheus: `sglang:queue_time_seconds`, `sglang:token_usage`; consider scaling out |
| **NaN logits** | Check model config, FP8 quantization; `--dump-tensor-path` for debugging |
| **Slow requests stuck** | Check `SGLANG_REQ_RUNNING_TIMEOUT`; increase timeout or investigate scheduler |
| **Memory leak** | Set `SGLANG_EMPTY_CACHE_INTERVAL=3600` for periodic cache purge |
| **Grammar timeout** | Set `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` higher or simplify grammar |

---

## 12. Operations Verdict

### Overall Ops Readiness: B (Production-capable with orchestration)

SGLang provides **strong operational foundations** for a GPU inference engine:

**Strengths:**
1. **Comprehensive observability** — 50+ Prometheus metrics, OpenTelemetry tracing with Gen-AI semantics, pre-built Grafana dashboard, profiling infrastructure
2. **Multi-platform Docker** — NVIDIA, AMD, Intel (XPU/Xeon), Ascend NPU, ARM64; separated dev/runtime images
3. **Mature CI/CD** — 74 workflows, 8-platform testing, dynamic partitioning, automated releases
4. **Good health checking** — Lightweight `/health` + deep `/health_generate` + comprehensive `/v1/loads`
5. **Graceful shutdown** — SIGTERM handling, in-flight request draining
6. **Rich performance tuning** — ~30 env vars + ~20 CLI flags directly affecting performance
7. **Production monitoring example** — Ready-to-use Prometheus + Grafana + Jaeger Docker Compose stack

**Critical Gaps (blockers for internet-facing production without additional tooling):**
1. **No authentication by default** — `--api-key` is opt-in; must be configured manually
2. **No rate limiting** — vulnerable to resource exhaustion without API gateway
3. **No TLS** — defer to reverse proxy (documented but not automated)
4. **No production K8s manifests** — test-only; production Helm charts/kustomize needed
5. **No HPA/scaling automation** — autoscaling must be built custom
6. **No SLO/SLA tracking** — metrics collected but no alerting thresholds defined

**Recommended Production Architecture:**

```
Internet → [TLS Terminator (nginx/Caddy)] → [API Gateway (auth, rate limit)] → [SGLang Gateway (routing)] → [SGLang Server(s)]
                                                                                        ↓
                                                                    ┌───────────────────┼───────────────────┐
                                                                    ↓                   ↓                   ↓
                                                              [Prometheus]        [Jaeger/OTel]       [Grafana]
```

### Production Readiness by Environment

| Environment | Readiness | Notes |
|---|---|---|
| **Research/single-user** | ✅ Ready | `pip install sglang && sglang serve --model-path ...` |
| **Internal team serving** | ✅ Ready | Add `--api-key`, deploy behind nginx |
| **Production (behind API gateway)** | ✅ Ready | Add auth/rate-limit/TLS at gateway layer |
| **Internet-facing (standalone)** | ❌ Not ready | Missing built-in auth, rate limiting, TLS |
| **Multi-tenant SaaS** | ⚠️ Needs work | Missing tenant quotas, billing, isolation |
| **Compliance (SOC2/HIPAA)** | ❌ Not ready | Missing audit logging, encryption at rest, access controls |

Evidence for all claims: See sections above with `Evidence:` annotations throughout.

