# API, SDK, CLI And Contracts

## 1. Public Surface Summary

| Surface | Count | Source | Stability | Evidence |
|---|---|---|---|---|
| **HTTP API routes** | ~40+ | http_server.py (native + OpenAI + Ollama + Anthropic + SageMaker + Vertex AI) | Stable (OpenAI-compatible) | http_server.py route decorators |
| **CLI commands** | 3 (serve, generate, version) + killall | cli/main.py, cli/generate.py, cli/killall.py | Stable | pyproject.toml:173-174 |
| **Python SDK (Frontend)** | sglang.lang API | lang/api.py, lang/backend/ | Stable (documented) | sglang.lang public module |
| **Python SDK (Engine)** | Engine.generate(), encode(), etc. | entrypoints/EngineBase.py | Stable (abstract base) | EngineBase.py |
| **Plugin API** | setuptools entry_points | plugins/__init__.py | Stable | plugins/__init__.py interface |
| **Attention Backend API** | register_attention_backend() | layers/attention/attention_registry.py | Internal (can change) | Registry-based |
| **Model API** | EntryClass convention | models/registry.py | Internal (auto-discovered) | Registry-based |
| **Spec Algorithm API** | register_algorithm() | speculative/spec_registry.py | Internal | Registry-based |
| **gRPC API** | Protobuf-defined | grpc/, proto/, rust/ | Beta | External smg-grpc-servicer package |

## 2. HTTP / GraphQL / RPC / WebSocket / gRPC

### 2.1 Native API Endpoints (http_server.py)

| Method | Path | Purpose | Params | Auth | Evidence |
|---|---|---|---|---|---|
| POST | `/generate` | Text generation | GenerateReqInput body | None | http_server.py |
| POST | `/encode` | Text embedding | EmbeddingReqInput body | None | http_server.py |
| POST | `/classify` | Reward/classification scoring | RewardReqInput body | None | http_server.py |
| GET | `/health` | Health check | None | None | http_server.py |
| GET | `/health_generate` | Deep health check (runs inference) | None | None | http_server.py |
| GET | `/get_server_info` | Server configuration info | None | None | http_server.py |
| GET | `/get_weights_by_name` | Query model weights | ?name=weight.path | None | http_server.py |
| POST | `/update_weights_from_disk` | Hot-reload weights from disk | UpdateWeightFromDiskReqInput | None | http_server.py |
| POST | `/update_weights_from_tensor` | Hot-reload weights from memory | Binary tensor data | None | http_server.py |
| POST | `/flush_cache` | Clear KV cache | None | None | http_server.py |
| POST | `/start_profile` | Start PyTorch profiler | StartProfileReqInput | None | http_server.py |
| POST | `/stop_profile` | Stop PyTorch profiler | StopProfileReqInput | None | http_server.py |
| POST | `/open_session` | Start multi-turn session | OpenSessionReqInput | None | http_server.py |
| POST | `/close_session` | End multi-turn session | CloseSessionReqInput | None | http_server.py |
| GET | `/metrics` | Prometheus metrics | None | None | http_server.py |

### 2.2 OpenAI-Compatible Endpoints

| Method | Path | Purpose | Compatibility |
|---|---|---|---|
| POST | `/v1/completions` | Text completions | OpenAI Completions API |
| POST | `/v1/chat/completions` | Chat completions | OpenAI Chat API |
| POST | `/v1/embeddings` | Text embeddings | OpenAI Embeddings API |
| GET | `/v1/models` | List available models | OpenAI Models API |
| POST | `/v1/responses` | Responses API (newer) | OpenAI Responses API |

Evidence: http_server.py `OpenAIServing` class, openai/ subdirectory adapters

### 2.3 Ollama-Compatible Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generate` | Ollama-style text generation |
| POST | `/api/chat` | Ollama-style chat |
| POST | `/api/tags` | List models |
| POST | `/api/pull` | Pull model (stub) |

Evidence: http_server.py `OllamaServing`, ollama/ subdirectory

### 2.4 Anthropic-Compatible Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/messages` | Anthropic Messages API |

Evidence: entrypoints/anthropic/ directory

### 2.5 SageMaker / Vertex AI Compatibility

Endpoints for AWS SageMaker and Google Vertex AI hosting environments.
Evidence: http_server.py route handlers

### 2.6 gRPC API
- **Protocol**: Defined in proto/ and rust/sglang-grpc/
- **Implementation**: Python binding via PyO3 to Rust gRPC server (`smg-grpc-servicer`)
- **Entry**: `serve_grpc(server_args)` in entrypoints/grpc_server.py
- **Thin wrapper**: Delegates heavy lifting to external `smg-grpc-servicer` package

## 3. CLI Commands

| Command | Flags | Input | Output | Exit Codes | Evidence |
|---|---|---|---|---|---|
| `sglang serve` | --model-path, --port, --tp-size, --mem-fraction-static, etc. (hundreds) | Model path/URL | Running HTTP server | 0 success, 1 failure | cli/serve.py, server_args.py |
| `sglang generate` | --model-path, --prompt, --sampling-params, --image-data | Prompt text/image | Generated text to stdout | 0 success, 1 failure | cli/generate.py |
| `sglang version` | None | None | Version string + git hash | 0 | cli/main.py:7-9 |
| `killall_sglang` | None | None | Kills all sglang processes | 0 success, 1 no processes | cli/killall.py |

### Key serve flags (subset of ~200+)
```
--model-path PATH          # Model identifier (HF, local, S3)
--port 30000               # HTTP listen port
--host 127.0.0.1           # Bind address
--tp-size 1                # Tensor parallelism size
--ep-size 1                # Expert parallelism size
--pp-size 1                # Pipeline parallelism size
--dp-size 1                # Data parallelism size
--mem-fraction-static 0.9  # GPU memory fraction for KV cache
--max-total-tokens 16384   # Max tokens per request
--context-length 8192      # Max context length
--schedule-policy fcfs     # Scheduling policy (fcfs, lpm, dfs-weight)
--attention-backend auto   # Attention backend
--speculative-algorithm NONE # Spec decoding (EAGLE, NGRAM, etc.)
--disaggregation-mode null  # Prefill-decode disaggregation
--quantization none        # Quantization (fp8, awq, gptq, etc.)
--enable-lora              # Enable LoRA adapters
--json-schema STR          # JSON schema for structured output
--log-level info            # Logging level
--api-key KEY              # API key for authentication
--served-model-name NAME    # Model name in API responses
--chat-template STR        # Override chat template
```

Evidence: server_args.py (7,950 lines of CLI argument definitions)

## 4. SDK Public Exports

### 4.1 Frontend DSL (sglang.lang)

| Export | Type | Signature | Purpose | Example | Evidence |
|---|---|---|---|---|---|
| `sglang.lang.Runtime` | Class | `Runtime(backend=...)` | Execute SGLang programs on a backend | `runtime = sglang.Runtime(model_path="meta-llama/Llama-3-8B")` | lang/api.py |
| `sglang.lang.choices` | Module | `sglang.gen()`, `sglang.select()`, `sglang.regex()` | Constrain generation via DSL | `s = sglang.gen("name", regex=r"[A-Z][a-z]+")` | lang/choices.py |
| `sglang.lang.interpreter` | Module | `run_program(prog, backend)` | Execute SGLang programs | Internal use | lang/interpreter.py |
| `sglang.lang.ir` | Module | `SglSamplingParams`, `SglExpr` | IR representations | Internal use | lang/ir.py |
| `sglang.lang.chat_template` | Module | `get_chat_template_by_model_path()` | Auto-detect chat templates | `get_chat_template_by_model_path("mistralai/Mistral-7B")` | lang/chat_template.py |
| `sglang.lang.tracer` | Module | `tracing_context()` | Execution tracing for debugging | Context manager | lang/tracer.py |

### 4.2 Engine API (sglang.srt)

| Export | Type | Signature | Purpose | Evidence |
|---|---|---|---|---|
| `Engine` | Class | `Engine(**server_args)` | Programmatic inference engine | engine.py:174 |
| `Engine.generate()` | Method | `generate(prompt, sampling_params, ...)` | Synchronous generation | EngineBase.py:13-38 |
| `Engine.async_generate()` | Method | `async_generate(prompt, ...)` | Async generation | engine.py |
| `Engine.encode()` | Method | `encode(texts, ...)` | Get embeddings | engine.py |
| `Engine.flush_cache()` | Method | `flush_cache()` | Clear KV cache | EngineBase.py:41-43 |
| `Engine.shutdown()` | Method | `shutdown()` | Graceful shutdown | EngineBase.py:74-76 |
| `Engine.update_weights_from_tensor()` | Method | `update_weights_from_tensor(named_tensors, ...)` | In-memory weight update | EngineBase.py:47-53 |

### 4.3 Public Python Functions

| Export | File | Purpose |
|---|---|---|
| `launch_server(server_args)` | launch_server.py | Launch HTTP server programmatically |
| `run_server(server_args)` | launch_server.py | Launch appropriate server mode |

### 4.4 Package Entry Points

```toml
[project.scripts]
sglang = "sglang.cli.main:main"
killall_sglang = "sglang.cli.killall:main"
```
Evidence: pyproject.toml:172-174

## 5. Plugin / Tool / Config API

### 5.1 Platform Plugin Contract
```python
# Entry point group: "sglang.srt.platforms"
# Contract: callable() -> SRTPlatform subclass
# Selection: SGLANG_PLATFORM env var or auto-detect
```
Evidence: plugins/__init__.py:28

### 5.2 General Plugin Contract
```python
# Entry point group: "sglang.srt.plugins"
# Contract: callable() with side effects (register hooks, replace classes)
# Selection: SGLANG_PLUGINS env var (comma-separated whitelist)
```
Evidence: plugins/__init__.py:29,51-55

### 5.3 Hook System
```python
# HookRegistry allows plugins to inject code before/after functions
HookRegistry.register_hook(target_function, hook_function, position="before"/"after")
HookRegistry.apply_hooks()  # called after all plugins load
```
Evidence: plugins/hook_registry.py

### 5.4 Attention Backend Contract
```python
@register_attention_backend("backend_name")
def create_backend(runner: ModelRunner) -> AttentionBackend:
    # Must return subclass of base_attn_backend.AttentionBackend
    ...
```
Evidence: layers/attention/attention_registry.py:23-29

### 5.5 Config API
- **Environment variables**: `SGLANG_*` prefix, defined via `Envs` descriptors in environ.py
- **CLI flags**: argparse-based, defined in server_args.py `ServerArgs` dataclass
- **Programmatic**: `Engine(port=30000, model_path="...", ...)`

## 6. Versioning And Compatibility

| Aspect | Status | Evidence |
|---|---|---|
| **API versioning** | No explicit version in API paths (no `/v1/` prefix except OpenAI compat) | http_server.py routes |
| **OpenAI compatibility** | Follows OpenAI API conventions | openai/ adapters |
| **Breaking changes** | No formal deprecation policy found | No DEPRECATION.md |
| **Package version** | Dynamic via setuptools_scm (git tags) | pyproject.toml:205-209 |
| **Config backward compat** | ServerArgs may add/remove fields between versions | server_args.py comment: "new args are backwards-compatible" |
| **Model compatibility** | Auto-discovery; new HF architectures work automatically | models/registry.py |
| **Plugin compatibility** | entry_points interface — plugins must track SGLang version | plugins/__init__.py |

### Versioning Risks
- **No semantic versioning guarantee**: setuptools_scm uses git tags; breaking changes possible without major version bump
- **Rapid evolution**: 12,649 commits, fast-moving codebase — API surface may shift
- **server_args.py volatility**: New flags added frequently; old flags may change defaults

## 7. Minimal Invocation Examples

### 7.1 CLI: Start Server
```bash
sglang serve --model-path meta-llama/Llama-3.2-3B-Instruct --port 30000
```
Evidence: README, server_args.py (not executed — no GPU)

### 7.2 CLI: Generate Text
```bash
sglang generate --model-path meta-llama/Llama-3.2-3B-Instruct --prompt "Hello, world!"
```
Evidence: cli/generate.py (not executed)

### 7.3 HTTP API: Generate
```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "sampling_params": {"max_new_tokens": 100}}'
```
Evidence: http_server.py route (not executed)

### 7.4 Python SDK: Engine
```python
from sglang import Engine
engine = Engine(model_path="meta-llama/Llama-3.2-3B-Instruct")
result = engine.generate("Hello, world!", {"max_new_tokens": 100})
```
Evidence: EngineBase.py, engine.py (not executed)

### 7.5 Python SDK: Frontend DSL
```python
import sglang as sgl

@sgl.function
def gen_character(s):
    s += "Generate a character name: "
    s += sgl.gen("name", regex=r"[A-Z][a-z]+", stop="\n")

runtime = sgl.Runtime(model_path="meta-llama/Llama-3.2-3B-Instruct")
runtime.run(gen_character)
```
Evidence: lang/api.py (conceptual, not executed)

### 7.6 OpenAI-Compatible API
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```
Evidence: openai/ adapters (not executed)

## 8. Contract Risks And Gaps

| Risk | Severity | Description | Mitigation |
|---|---|---|---|
| **No API versioning** | Medium | Breaking changes to HTTP API not signaled | Pin SGLang version in production; test before upgrade |
| **No formal deprecation policy** | Medium | Fields may be removed without warning | Monitor release notes and changelogs |
| **ZMQ IPC is internal contract** | Low | Message formats can change between versions | Always match Engine + Scheduler + Detokenizer versions |
| **Plugin API is undocumented** | Medium | HookRegistry, entry_points interfaces not in official docs | Read plugins/__init__.py source |
| **No API key enforcement by default** | High | `--api-key` flag exists but default is no auth | Always set `--api-key` in production |
| **Limited error codes** | Low | HTTP errors use generic 400/500; no SGLang-specific error taxonomy | Parse error messages from response body |
| **Streaming response format** | Low | SSE format may differ between native and OpenAI endpoints | Test with actual client libraries |
| **Multi-modal API evolving** | Medium | Image/video/audio input format may change between versions | Check multimodal_gen/ and io_struct.py changes |
