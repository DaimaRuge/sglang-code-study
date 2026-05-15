# Runtime And Execution Lifecycle

## 1. Runtime Entry Points

| Mode | Command | Entry File | Symbol | Evidence |
|---|---|---|---|---|
| Default (HTTP) | `sglang serve` | python/sglang/cli/main.py → cli/serve.py → launch_server.py | `run_server()` → `launch_server(server_args)` | launch_server.py:47-50 |
| gRPC | `sglang serve --grpc-mode` | python/sglang/cli/main.py → cli/serve.py → launch_server.py | `run_server()` → `serve_grpc(server_args)` | launch_server.py:33-35 |
| Ray | `sglang serve --use-ray` | python/sglang/cli/main.py → cli/serve.py → launch_server.py | `run_server()` → Ray `launch_server(server_args)` | launch_server.py:36-45 |
| Encoder-only | `sglang serve --encoder-only` | python/sglang/cli/main.py → cli/serve.py → launch_server.py | `run_server()` → encoder `launch_server()` | launch_server.py:17-22 |
| Generate | `sglang generate` | python/sglang/cli/generate.py | `generate` subcommand | pyproject.toml:173 |
| Version | `sglang version` | python/sglang/cli/main.py | `version()` | pyproject.toml:173 |
| Kill All | `killall_sglang` | python/sglang/cli/killall.py | `main()` | pyproject.toml:174 |

### Process Architecture
The SGLang runtime uses a **three-process pipeline**:

```
Main Process (PID=N)          Scheduler Subprocess (PID=N+1)      Detokenizer Subprocess (PID=N+2)
┌──────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐
│ Engine               │     │ Scheduler               │     │ DetokenizerManager       │
│ TokenizerManager     │ ZMQ │ ModelRunner (GPU)       │ ZMQ │ Tokenizer (CPU)          │
│ HTTP Server (FastAPI)│────▶│ RadixCache              │────▶│ Response Assembly        │
│ TemplateManager      │◀────│ SpeculativeWorker       │◀────│ Chat Template Formatting │
│ asyncio event loop   │     │ GrammarManager          │     │                          │
│ (main thread)        │     │ synchronous event loop  │     │ asyncio event loop       │
└──────────────────────┘     └─────────────────────────┘     └──────────────────────────┘
```

Evidence: engine.py:174-186 (docstring describing three components), engine.py:84-85 (mp.Process), engine.py:47 (ZMQ import)

## 2. Startup Chain

```text
CLI: sglang serve
  → sglang.cli.main:main()                        [cli/main.py:12]
  → sglang.cli.serve:serve()                      [cli/serve.py]
  → sglang.launch_server:run_server(server_args)  [launch_server.py:15]
  → Engine.__init__(**server_args)                [engine.py:195]
    → load_plugins()                               [plugins/__init__.py:103]
    → ServerArgs.__post_init__()                   [server_args.py] (validation)
    → _set_envs_and_config()                       [engine.py] (NCCL, ulimit, signal handlers)
    → PortArgs.init_new()                          [port args allocation]
    → _launch_subprocesses()                       [engine.py:678]
      → _launch_scheduler_processes()              [mp.Process for each TP rank]
        → run_scheduler_process()                  [managers/scheduler.py]
          → load_plugins()                          [scheduler.py:3999]
          → Scheduler.__init__()                    [scheduler.py:340]
          → Scheduler.event_loop_normal/overlap()   [scheduler.py:1537]
      → mp.Process(run_detokenizer_process)        [detokenizer subprocess]
      → init_tokenizer_manager()                   [tokenizer_manager.py]
      → TemplateManager.initialize_templates()     [template detection]
    → _wait_for_scheduler_ready()                  [pipe-based readiness check]
    → SubprocessWatchdog.start()                   [liveness monitoring]
    → Engine.__init__() finalize
  → HTTP: launch_server(server_args)               [http_server.py]
    → _GlobalState setup                           [http_server.py:190]
    → FastAPI app creation + routes                [http_server.py]
    → _wait_and_warmup()                            [warmup.py]
    → uvicorn.run() / granian.serve()             [http_server.py]
  → Runtime Ready (accepting requests)
```

Evidence: engine.py:174-700+, http_server.py:190, scheduler.py:340-3999

## 3. Initialization Steps

| Order | Step | File/Symbol | Purpose | Failure Mode |
|---|---|---|---|---|
| 1 | **Plugin Loading** | plugins/__init__.py:103 `load_plugins()` | Discover and register hardware platforms, general plugins via setuptools entry_points | Plugin load failure → logged, continues |
| 2 | **ServerArgs Parsing** | server_args.py `ServerArgs.__init__()` | Parse CLI args, env vars, validate | Invalid args → argparse error, exit |
| 3 | **Environment Setup** | engine.py `_set_envs_and_config()` | Set NCCL env vars, ulimit, signal handlers | Permission denied → partial setup |
| 4 | **Port Allocation** | server_args.py `PortArgs.init_new()` | Allocate ZMQ IPC ports, HTTP ports | Port conflict → retry or fail |
| 5 | **MP Start Method** | engine.py line ~210 | Set multiprocessing start method to "spawn" | Windows: already "spawn" |
| 6 | **Scheduler Fork** | engine.py `_launch_scheduler_processes()` | Fork scheduler subprocess(es) via mp.Process | CUDA init in child → crash |
| 7 | **Detokenizer Fork** | engine.py line ~680 | Fork detokenizer subprocess | OOM → crash |
| 8 | **Tokenizer Init** | tokenizer_manager.py `TokenizerManager.__init__()` | Load HF tokenizer, create IPC sockets | Model not found → FileNotFoundError |
| 9 | **Template Detection** | template_manager.py `initialize_templates()` | Detect chat template, auto-configure reasoning/tool parsers | Unknown template → warning |
| 10 | **CUDA Graph Capture** | model_runner.py `init_device_graphs()` | Capture CUDA graphs for batch sizes | OOM → fallback to eager mode |
| 11 | **Memory Pool Init** | model_runner_kv_cache_mixin.py `init_memory_pool()` | Allocate KV cache, radix cache | Insufficient GPU memory → OOM |
| 12 | **Attention Backend Init** | model_runner.py `init_attention_backend()` | Select and warm up attention backend | Incompatible backend → fallback |
| 13 | **Warmup** | warmup.py `_wait_and_warmup()` | Send dummy request to initialize all CUDA kernels | Warmup timeout → retry or fail |
| 14 | **HTTP Server Start** | http_server.py `uvicorn.run()` | Start accepting HTTP requests | Port in use → AddressInUseError |

### Detailed Initialization: Scheduler Process

```text
run_scheduler_process(server_args, port_args, gpu_id, tp_rank, ...)
  → load_plugins()                                    [plugin hooks before Scheduler init]
  → Scheduler.__init__(server_args, port_args, ...)   [scheduler.py:340]
    → Set rank attributes (tp_rank, pp_rank, etc.)   [scheduler.py:356-368]
    → Parse spec algorithm from string                [scheduler.py:389-391]
    → Initialize distributed environment              [scheduler.py:669-725]
      → init_distributed_environment()                 [parallel_state.py:1668]
      → initialize_model_parallel()                    [parallel_state.py:1755]
    → Create TpModelWorker                            [scheduler.py]
      → ModelRunner.__init__()                         [model_runner.py:329]
        → initialize()                                 [model_runner.py:535]
          → create_sampler()                           [model_runner.py:654]
          → load_model()                               [model_runner.py:655]
            → get_model_loader() → DefaultModelLoader  [model_loader/loader.py]
            → loader.load_model()                      [model_loader/loader.py:700]
              → _initialize_model() via ModelRegistry  [model_loader/loader.py:272]
              → model.load_weights(weight_iterator)    [per-model load_weights]
          → init_memory_pool()                         [kv cache allocation]
          → init_attention_backend()                   [attention backend selection]
          → init_device_graphs()                       [CUDA graph capture]
    → Create RadixCache                               [scheduler.py]
    → Create GrammarManager                           [scheduler.py]
    → Create speculative worker (if enabled)          [scheduler.py]
    → Create disaggregation manager (if enabled)      [scheduler.py]
    → Signal readiness via pipe                       [scheduler.py]
    → Enter event loop                                 [scheduler.py:1537]
```

### Detailed Initialization: Detokenizer Process

```text
run_detokenizer_process(server_args, port_args)
  → DetokenizerManager.__init__()
    → Load tokenizer (same vocab as main process)
    → Create ZMQ sockets to scheduler and tokenizer_manager
    → Start asyncio event loop for receiving/processing detokenization
```

## 4. Configuration Loading Order

| Priority | Source | File/Symbol | Example Keys | Risk |
|---|---|---|---|---|
| 1 (lowest) | **Code defaults** | server_args.py dataclass fields | `port=30000`, `tp_size=1`, `mem_fraction_static=0.9` | Defaults may not suit all hardware |
| 2 | **Environment variables** | environ.py `EnvField` descriptors | `SGLANG_TP_SIZE`, `SGLANG_ENABLE_FLASHINFER`, `SGLANG_ATTENTION_BACKEND` | Bypasses CLI validation |
| 3 | **CLI flags** | server_args.py argparse | `--tp-size 4`, `--model-path meta-llama/Llama-3-8B`, `--port 30000` | Overrides env vars |
| 4 | **Config file** | server_args_config_parser.py | `--config-file config.json` with JSON config | Format errors → silent failure |
| 5 | **Runtime override** | `Engine(port=30001)` direct kwargs | Programmatic API | Bypasses all validation |
| 6 (highest) | **Platform defaults** | platforms/ interface `apply_server_args_defaults()` | Platform-specific attention backend, compile settings | Hidden overrides |

### Configuration Environment Variables (Key Examples)

| Env Var | Purpose | Default | Source |
|---|---|---|---|
| `SGLANG_ATTENTION_BACKEND` | Override attention backend | None (auto-select) | environ.py |
| `SGLANG_ENABLE_FLASHINFER` | Enable FlashInfer kernels | True (if CUDA) | environ.py |
| `SGLANG_PLATFORM` | Select hardware platform | Auto-detect | plugins/__init__.py |
| `SGLANG_PLUGINS` | Comma-separated plugin whitelist | All installed | plugins/__init__.py:52 |
| `SGLANG_GRPC_MODE` | Enable gRPC server mode | False | environ.py |
| `SGLANG_RETURN_ORIGINAL_LOGPROB` | Return original token logprobs | False | environ.py |
| `SGLANG_ALLOW_OVERWRITE_WEIGHTS` | Allow weight overwrite in-memory | False | environ.py |
| `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` | Max recv per scheduler poll | Configurable | scheduler.py:396 |

Evidence: environ.py (entire file for EnvField descriptor system), server_args.py (entire file for CLI args), plugins/__init__.py:52

## 5. Plugin / Provider / Middleware Registration

### Plugin Registration (Before ServerArgs)
```
load_plugins()  # Called in Engine.__init__() and run_scheduler_process()
  ├── Discover platforms: entry_points(group="sglang.srt.platforms")
  │   ├── If SGLANG_PLATFORM set: load only that platform
  │   └── If auto-detect: load all, activate the one that reports ready
  ├── Discover general plugins: entry_points(group="sglang.srt.plugins")
  │   ├── If SGLANG_PLUGINS set: whitelist filter
  │   └── Load each plugin, exclude unselected platform dists
  └── HookRegistry.apply_hooks()
      └── Apply all registered hooks (idempotent — skips already-patched)
```

Evidence: plugins/__init__.py:103-141

### Attention Backend Registration
```python
ATTENTION_BACKENDS = {}

@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    # Creates FlashInferAttnBackend or FlashInferMLAAttnBackend
    ...

# Registered backends: flashinfer, fa3, fa4, flashmla, cutlass_mla, triton,
# trtllm_mha, trtllm_mla, aiter, nsa, dsv4, tokenspeed_mla, ascned,
# intel_amx, intel_xpu, dual_chunk_flash_attn, torch_native, flex_attention,
# hybrid, wave, tbo, flashinfer_mla
```

Evidence: layers/attention/attention_registry.py:23-29 (registry pattern), lines 31-400+ (registered backends)

### Model Auto-Registration
```python
# models/registry.py:131
ModelRegistry.register("sglang.srt.models")
# Auto-scans all .py files in the models/ directory
# Discovers classes via EntryClass attribute
```

### LoRA Backend Registration
```python
# lora/backend/lora_registry.py
LORA_BACKEND_REGISTRY = {
    "triton": TritonLoRABackend,
    "csgmv": ChunkedSgmvLoRABackend,
    "ascend": AscendLoRABackend,
    "torch_native": TorchNativeLoRABackend,
}
```

### Speculative Algorithm Registration
```python
# speculative/spec_registry.py
SpeculativeAlgorithm.register("MY_SPEC")(MyCustomSpecAlgo)
```

## 6. Logging And Error Handling Initialization

### Logging Setup
```text
engine.py:
  → configure_logger(server_args)              [utils/configure_logging.py]
    → logging.getLogger("sglang")
    → Set log level from --log-level flag
    → Set log format, file output if requested
    → setproctitle.setproctitle("sglang::engine")  [Process naming]
```
Evidence: engine.py imports `configure_logger`, utils/configure_logging.py

### Error Handling Architecture

| Layer | Error Type | Handling Strategy |
|---|---|---|
| HTTP Server | FastAPI exceptions | FastAPI exception handlers → HTTP 4xx/5xx |
| TokenizerManager | Tokenization error | Caught in `_tokenize_one_request()`, returned as error response |
| Scheduler | CUDA OOM, kernel crash | Subprocess crash → Watchdog detects → Engine restarts |
| ModelRunner | GPU errors (OOM, misconfig) | try/except in forward pass, logged, returned to scheduler |
| Detokenizer | Invalid token IDs | Filtered, logged, skip bad tokens |
| IPC (ZMQ) | Connection loss | Socket timeout + reconnection logic, Watchdog monitoring |
| Plugin loading | Import error, init failure | Caught in `load_plugins()`, logged, skipped |

### SubprocessWatchdog
```text
engine.py: SubprocessWatchdog
  → Monitors scheduler and detokenizer liveness via heartbeat
  → If subprocess dies: kill all related processes, attempt restart
  → Configurable timeout
```
Evidence: engine.py imports `SubprocessWatchdog` from utils/watchdog

## 7. Lifecycle Hooks And Graceful Shutdown

### Runtime Status Transitions
```
Starting → (warmup) → Up → (shutdown) → Stopping → Stopped
```

### Shutdown Sequence
```text
1. SIGTERM / SIGINT received
   → signal_handler in engine.py sets shutdown_event
2. HTTP server stops accepting new requests
   → uvicorn shutdown
3. Engine.shutdown()
   → TokenizerManager.shutdown(): flush pending requests
   → Scheduler subprocess: send shutdown signal via ZMQ
   → Detokenizer subprocess: send shutdown signal
   → Wait for subprocesses to exit (timeout)
   → Force kill if timeout
4. Cleanup
   → Release GPU memory
   → Close ZMQ sockets
   → Remove temporary files
   → Shutdown distributed process groups
5. Exit
```
Evidence: engine.py: EngineBase.shutdown() abstract method, engine.py signal handlers

### EngineBase Lifecycle Methods
```python
class EngineBase(ABC):
    generate()                    # Main inference entry
    flush_cache()                 # Clear KV cache
    update_weights_from_tensor()  # Hot-reload weights
    release_memory_occupation()   # Free GPU memory for other processes
    resume_memory_occupation()    # Reclaim freed memory
    shutdown()                    # Graceful shutdown
```
Evidence: srt/entrypoints/EngineBase.py:7-77

## 8. Runtime Verification Log

| Command | Result | Evidence/Output Summary | Confidence |
|---|---|---|---|
| `python -c "import sglang"` | Not attempted (no GPU/CUDA on Windows) | Would fail due to torch CUDA dependency | N/A (not run) |
| `python -m sglang.cli --help` | Not attempted | Would show serve, generate, version subcommands | Medium (from code) |
| `python -m sglang.cli serve --help` | Not attempted | Would show all ServerArgs options | Medium (from server_args.py) |
| `python -m sglang.cli version` | Not attempted | Would show setuptools_scm version | Medium (from code) |
| Full serve startup | **Cannot run** — requires NVIDIA GPU, CUDA, model weights | N/A | N/A |
| Minimal test run | **Cannot run** — tests require GPU | N/A | N/A |

### Static-Only Verification Methods Used

| Method | What It Verified | Reliability |
|---|---|---|
| Code tracing (engine.py, scheduler.py) | Initialization sequence, process architecture | High |
| Import analysis (4 exploration agents) | Module dependency graph, plugin registration | High |
| Configuration analysis (server_args.py, environ.py) | Config loading order, env var surface | High |
| Architecture-agent synthesis | Request flow: HTTP → Tokenizer → Scheduler → Detokenizer | High |
| pyproject.toml + dependency analysis | Build system, runtime dependencies | High |

## 9. Static-only Limitations

1. **GPU-specific initialization not verified**: CUDA graph capture, attention backend warmup, memory pool allocation all require GPU hardware
2. **IPC (ZMQ) not tested**: Message formats, serialization, timeout behavior inferred from code
3. **Multi-GPU distributed init not verified**: NCCL group creation, all-reduce setup requires multi-GPU
4. **Plugin loading not verified**: `entry_points()` discovery requires pip-installed packages
5. **Warmup request flow not verified**: The warmup sends a real inference request through the full pipeline
6. **Signal handling not verified**: SIGTERM/SIGINT behavior inferred from code patterns
7. **Docker/container paths not verified**: Dockerfile-based deployment untested
8. **Performance not measured**: No benchmarks can be run without GPU

### To Verify If GPU Becomes Available
1. Run `sglang serve --model-path <small-model> --port 30000`
2. Send a curl request: `curl http://localhost:30000/generate -d '{"text": "Hello"}'`
3. Check metrics: `curl http://localhost:30000/metrics`
4. Run the test suite: `cd test && python run_suite.py`
5. Check logs for the complete initialization sequence (all 14 steps from Section 3)
