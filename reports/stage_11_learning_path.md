# Learning Path And Code Reading Plan

## 1. Learner Profile And Goal

This plan targets four learner profiles:

| Profile | Background | Goal | Time/Week |
|---|---|---|---|
| **Beginner** | Python proficient, knows PyTorch basics, new to LLM serving | Run SGLang, understand request flow, make trivial changes | 5-10 hours |
| **Intermediate** | ML engineer, familiar with LLM inference, some CUDA knowledge | Add model support, debug performance, write tests, contribute | 10-15 hours |
| **Advanced** | Systems/GPU engineer, experienced with vLLM/TGI/TRT-LLM | Design new backends, optimize kernels, contribute major features | 15-20+ hours |
| **Production** | MLOps/platform engineer, focused on deployment | Deploy at scale, configure observability, harden security | 10-15 hours |

## 2. Prerequisite Knowledge Map

| Concept | Required Level | Why Needed | Suggested Resource |
|---|---|---|---|
| **Python async/await** | Intermediate | TokenizerManager event loop is asyncio-based; understanding `await`, `asyncio.Event`, `asyncio.Queue` is essential | Python official asyncio docs |
| **PyTorch `nn.Module`** | Intermediate | Every model is an `nn.Module`; weights loading, forward(), tensor operations | PyTorch tutorials |
| **CUDA / GPU concepts** | Intermediate | KV cache layout in GPU memory, CUDA streams for overlap, CUDA graphs | CUDA C++ Programming Guide (Ch 3) |
| **Transformer architecture** | Intermediate | Self-attention, KV cache, RoPE, GQA/MQA/MLA, MoE | "Attention Is All You Need", vLLM blog posts |
| **ZMQ / IPC** | Intermediate (for core dev) | Three-process subprocess communication via ZMQ PUSH/PULL/DEALER | ZMQ Guide (zguide.zeromq.org) |
| **NCCL / distributed computing** | Advanced | TP/PP/EP communication patterns, all-reduce, all-gather, all-to-all | NCCL documentation |
| **Triton (GPU kernel language)** | Advanced | Custom attention kernels, KV cache allocation kernels, MoE dispatch | Triton tutorials |
| **LLM serving concepts** | Beginner | Continuous batching, prefix caching, speculative decoding, chunked prefill | SGLang blog posts, vLLM paper |
| **FastAPI / HTTP** | Beginner (for API use) | HTTP endpoints, streaming responses (SSE), middleware | FastAPI docs |
| **HuggingFace transformers** | Beginner | Model config, tokenizers, weight loading from Hub | HuggingFace course |
| **Rust (optional)** | Advanced | gRPC server (optional component), model gateway | Rust book |

## 3. Code Reading Order

### Phase 1: Orientation (Day 1-2)

Read in this order — each file answers a specific question:

| Order | File/Directory | Focus | Expected Insight | Time |
|---|---|---|---|---|
| 1 | `README.md` | Project overview, key claims | What SGLang is, why it exists, performance claims, who maintains it | 15 min |
| 2 | `python/pyproject.toml` | Dependencies, entry points, build config | What packages SGLang depends on, what CLI commands exist (`sglang`, `killall_sglang`) | 10 min |
| 3 | `python/sglang/cli/main.py` | CLI entry point | How `sglang serve` dispatches to `run_server()`; only 10 lines of logic | 5 min |
| 4 | `python/sglang/launch_server.py` | Server launcher | How server_args flow into HTTP/gRPC/Ray/Encoder modes | 10 min |
| 5 | `python/sglang/srt/entrypoints/http_server.py` | HTTP routes (scan only) | What endpoints exist (`/generate`, `/v1/chat/completions`, `/health`, etc.); `_GlobalState` singleton pattern | 30 min |
| 6 | `python/sglang/srt/server_args.py` | Configuration flags (scan first 500 lines) | What configuration exists; how `ServerArgs` dataclass is structured; argparse generation | 30 min |

**Phase 1 checkpoint question:** "If I run `sglang serve --model-path meta-llama/Llama-3-8B --port 30000`, what process starts, what ports open, and what endpoints are available?"

### Phase 2: The Core Pipeline (Day 3-7)

This is where you understand the heart of SGLang. Read top-to-bottom in request flow order.

| Order | File | Key Sections | Question Answered | Time |
|---|---|---|---|---|
| 7 | `python/sglang/srt/managers/io_struct.py` | All `*ReqInput`, `*ReqOutput` classes | What data structures flow between processes? What does a client request look like on the wire? | 30 min |
| 8 | `python/sglang/srt/entrypoints/engine.py` | `Engine.__init__()`, `_launch_subprocesses()`, three-component docstring (lines 178-186) | How does Engine fork TokenizerManager, Scheduler, Detokenizer? What ZMQ sockets connect them? | 45 min |
| 9 | `python/sglang/srt/managers/tokenizer_manager.py` | `generate_request()`, `_tokenize_one_request()`, `handle_loop()` (line 1638) | How does a text prompt become token IDs? How does the async response flow work? | 1 hour |
| 10 | `python/sglang/srt/managers/scheduler.py` | `event_loop_normal()` (line 1537), `event_loop_overlap()` (line 1564), `get_next_batch_to_run()` (line 2485), `process_batch_result()` | How does the scheduler merge requests into batches? How does CPU/GPU overlap work? What is the schedule policy? | 2 hours |
| 11 | `python/sglang/srt/managers/schedule_batch.py` | `Req` (line 578), `ScheduleBatch` (line 1381) | What state does a request carry? How is a batch of requests represented? | 1 hour |
| 12 | `python/sglang/srt/model_executor/model_runner.py` | `forward()`, `load_model()` (line 1371), CUDA graph capture | How does a batch get executed on GPU? How are models loaded? What is captured in CUDA graphs? | 2 hours |
| 13 | `python/sglang/srt/layers/radix_attention.py` | `RadixAttention.forward()` | How does every model's attention layer dispatch to the selected attention backend? | 30 min |
| 14 | `python/sglang/srt/mem_cache/radix_cache.py` | `match_prefix()` (line 360), `insert()` (line 420), `evict()` (line 560) | How does automatic prefix caching work? How are KV cache indices shared across requests? | 1 hour |
| 15 | `python/sglang/srt/sampling/sampler.py` | `Sampler.forward()` | How do logits become token IDs? Where do grammar constraints apply? | 30 min |

**Phase 2 checkpoint question:** "Trace a request from `POST /generate` through tokenization → scheduling → GPU forward → sampling → detokenization → response. What data transformations happen at each stage?"

### Phase 3: Deep Dives (Day 8-14)

Pick deep-dive topics based on your goals:

#### Track A: Model Developer (adding model support)

| Order | File | Focus | Time |
|---|---|---|---|
| A1 | `python/sglang/srt/models/registry.py` | How `ModelRegistry` auto-discovers models via `pkgutil.iter_modules` looking for `EntryClass` | 30 min |
| A2 | `python/sglang/srt/models/llama.py` | Canonical model implementation: embedding → layers → norm → lm_head; `load_weights()` pattern | 1 hour |
| A3 | `python/sglang/srt/configs/model_config.py` | How HF config is parsed into SGLang's internal `ModelConfig`; attention architecture detection | 45 min |
| A4 | `python/sglang/srt/model_loader/loader.py` | How `DefaultModelLoader.load_model()` instantiates + loads weights | 1 hour |
| A5 | 3-5 model files in `models/` with different attention patterns (MHA, GQA, MLA, NSA, etc.) | Compare patterns: what varies, what's constant | 2 hours |

#### Track B: Systems/Performance Engineer

| Order | File | Focus | Time |
|---|---|---|---|
| B1 | `python/sglang/srt/model_executor/cuda_graph_runner.py` | CUDA graph capture (`capture()`, line 850), replay (`replay()`, line 1307), batch size padding | 1.5 hours |
| B2 | `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | How batch results are processed: token extraction, grammar state update, cache insertion | 1 hour |
| B3 | `python/sglang/srt/managers/overlap_utils.py` | `FutureMap` — how overlap scheduling resolves future token IDs with a circular buffer (line 121) | 45 min |
| B4 | `python/sglang/srt/layers/attention/flashinfer_backend.py` | How FlashInfer paged attention integrates: `init_forward_metadata`, `forward_decode`, `forward_extend` | 1.5 hours |
| B5 | `python/sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool`, `MHATokenToKVPool` — how memory is laid out and addressed | 1 hour |
| B6 | `python/sglang/srt/managers/schedule_policy.py` | `PrefillAdder` (line 407) — token budget estimation, scheduling algorithm (fcfs/lpm/dfs-weight) | 1 hour |

#### Track C: Extension/Plugin Developer

| Order | File | Focus | Time |
|---|---|---|---|
| C1 | `python/sglang/srt/plugins/__init__.py` | `load_plugins()` — setuptools entry_points discovery; `HookRegistry.apply_hooks()` | 30 min |
| C2 | `python/sglang/srt/layers/attention/attention_registry.py` | `@register_attention_backend()` decorator — how 28 backends register themselves | 20 min |
| C3 | `python/sglang/srt/speculative/spec_registry.py` | `@register_algorithm()` — how to add a custom spec algorithm | 20 min |
| C4 | `python/sglang/srt/constrained/base_grammar_backend.py` | Grammar backend interface — how to add a new grammar engine | 30 min |
| C5 | `python/sglang/srt/disaggregation/base/conn.py` | Transfer backend interface — how to add a new KV transfer backend (RDMA, TCP, etc.) | 30 min |

#### Track D: Production/MLOps Engineer

| Order | File | Focus | Time |
|---|---|---|---|
| D1 | `python/sglang/srt/observability/metrics_collector.py` | All 50+ Prometheus metrics — what each means, what labels exist | 1.5 hours |
| D2 | `examples/monitoring/` | Prometheus + Grafana + Jaeger Docker Compose setup | 30 min |
| D3 | `python/sglang/srt/observability/trace.py` | OpenTelemetry tracing: `TraceReqContext`, W3C propagation, Gen-AI semantic conventions | 1 hour |
| D4 | `docker/Dockerfile` | Multi-stage Docker build: base → torch_deps → framework → runtime | 1 hour |
| D5 | `python/sglang/srt/environ.py` | 80+ `SGLANG_*` env vars organized by category; `EnvField` descriptor system | 1 hour |

### Phase 4: Quality And Contribution (Day 15-21)

| Order | File/Directory | Focus | Time |
|---|---|---|---|
| 16 | `test/srt/` | Browse test structure: how tests are organized, what fixtures exist, `@parameterized.expand` usage | 2 hours |
| 17 | `.github/workflows/pr-test.yml` | How PR CI works: stages, gates, hardware matrix | 1 hour |
| 18 | `.pre-commit-config.yaml` | Lint checks: ruff, isort, black, codespell, clang-format | 15 min |
| 19 | `test/run_suite.py` | How the test suite is orchestrated: partitioning, stage selection | 30 min |
| 20 | `python/sglang/check_env.py` | Environment verification: CUDA, PyTorch, GPU info, package versions | 20 min |

## 4. Hands-on Labs

### Lab 1: Run SGLang (Beginner)

| Aspect | Detail |
|---|---|
| **Goal** | Start SGLang serving a small model and send requests |
| **Prerequisites** | NVIDIA GPU (≥8GB VRAM), CUDA 12.x, Docker or pip install |
| **Steps** | 1. `pip install "sglang[all]"` (or use Docker) 2. `sglang serve --model-path meta-llama/Llama-3.2-3B-Instruct --port 30000` 3. In another terminal: `curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": "Explain quantum computing in one sentence"}]}'` 4. Try the OpenAI-compatible endpoint: same curl with `/v1/completions` 5. Check health: `curl http://localhost:30000/health` and `/health_generate` |
| **Validation** | Server starts without errors, API returns generated text, `/health` returns 200 |
| **Difficulty** | Easy |

### Lab 2: Trace A Request (Beginner → Intermediate)

| Aspect | Detail |
|---|---|
| **Goal** | Trace a single request through the entire pipeline using debug logs |
| **Steps** | 1. Start server with `--log-level debug --log-requests --log-requests-level 3` 2. Send a single request 3. Follow the log output: find the tokenization, scheduling ("get_new_batch_prefill"), forward ("run_batch"), sampling, detokenization 4. Draw the request's path through the three processes 5. Identify which log lines come from TokenizerManager vs Scheduler vs Detokenizer |
| **Validation** | Can identify: (a) when the request was tokenized, (b) when it was scheduled into a batch, (c) when it generated each token, (d) when it was detokenized and returned |
| **Difficulty** | Easy-Medium |

### Lab 3: Understand The Scheduler (Intermediate)

| Aspect | Detail |
|---|---|
| **Goal** | Understand how the scheduler selects and batches requests |
| **Steps** | 1. Read `scheduler.py` `get_next_batch_to_run()` (line 2485) 2. Add temporary print statements showing: batch size, forward mode (EXTEND/DECODE), which RIDs are in each batch 3. Send 5 concurrent requests with different prompt lengths 4. Observe: (a) Which requests are batched together and why? (b) When does the mode switch from EXTEND to DECODE? (c) How does the RadixCache match_prefix affect batching? 5. Try with `--schedule-policy lpm` vs default `fcfs` — observe priority differences |
| **Validation** | Can explain why specific requests are grouped together and predict batching behavior for a given set of requests |
| **Difficulty** | Medium |

### Lab 4: Add A New Model Architecture (Intermediate → Advanced)

| Aspect | Detail |
|---|---|
| **Goal** | Add support for a new HuggingFace model architecture (e.g., a fine-tuned LLaMA variant with custom layers) |
| **Steps** | 1. Read `models/llama.py` as reference 2. Create `models/my_model.py` 3. Implement `MyModelForCausalLM.__init__()` — define layers matching HF config 4. Implement `load_weights()` — map HF weight names to SGLang layer names 5. Set `EntryClass = MyModelForCausalLM` at module level 6. Test: `sglang serve --model-path /path/to/my-model --port 30000` 7. Verify: send generate request, compare output with HF transformers output |
| **Validation** | Model loads without error, generates text, logprobs match HF transformers reference within 1e-3 |
| **Difficulty** | Medium-Hard |

### Lab 5: Add A Test (Intermediate)

| Aspect | Detail |
|---|---|
| **Goal** | Write and run a new integration test |
| **Steps** | 1. Find an untested edge case (see Stage 8 coverage blind spots) 2. Create `test/srt/test_my_feature.py` 3. Use existing fixtures: `engine_fixture`, `scheduler_fixture` 4. Write a test that: sends a request, validates the response shape/content 5. Optionally use `@parameterized.expand` for multiple configs 6. Run: `pytest test/srt/test_my_feature.py -xvs` |
| **Validation** | Test passes; test fails meaningfully when feature breaks |
| **Difficulty** | Medium |

### Lab 6: Profile And Optimize (Advanced)

| Aspect | Detail |
|---|---|
| **Goal** | Profile the model forward pass and identify a bottleneck |
| **Steps** | 1. Start server with `--enable-metrics` 2. Run benchmark with `test/bench_one_batch_server_internal.py` 3. Call `POST /start_profile` with `num_steps=10`, `activities=["CPU","GPU"]` 4. Call `POST /stop_profile` and download trace 5. Open in Chrome `chrome://tracing` 6. Identify: (a) What % of time is spent in attention vs MLP? (b) Are there GPU idle gaps? (c) What kernel is the bottleneck? |
| **Validation** | Can identify the top-3 time-consuming operations and suggest a concrete optimization |
| **Difficulty** | Hard |

### Lab 7: Deploy With Monitoring (Production)

| Aspect | Detail |
|---|---|
| **Goal** | Deploy SGLang with full observability stack |
| **Steps** | 1. Start server with `--enable-metrics` 2. `docker compose -f examples/monitoring/docker-compose.yaml up` 3. Open Grafana at `http://localhost:3000` (default admin/admin) 4. Import the SGLang dashboard 5. Send requests and observe live metrics 6. Check `/v1/loads` for detailed load info 7. Optionally: set up OTEL tracing with `pip install sglang[tracing]` |
| **Validation** | Grafana dashboard shows live metrics; can identify TTFT, throughput, cache hit rate, queue depth |
| **Difficulty** | Medium |

## 5. Modify A Small Feature

### Exercise A: Add A Custom Health Endpoint (Easy, ~30 min)

| Aspect | Detail |
|---|---|
| **Change** | Add `GET /my-health` that returns `{"status": "ok", "gpu_utilization": X}` |
| **Files** | `http_server.py` — add route handler |
| **Key code** | Read `sglang:utilization` gauge from Prometheus metrics; return as JSON |
| **Test** | `curl http://localhost:30000/my-health` returns valid JSON |
| **Expected Result** | Custom health endpoint returns GPU utilization alongside status |

### Exercise B: Add A Simple Scheduling Policy (Medium, ~4 hours)

| Aspect | Detail |
|---|---|
| **Change** | Add `shortest-first` scheduling policy that prioritizes requests with fewer remaining tokens |
| **Files** | `schedule_policy.py` — add policy class; `server_args.py` — add `shortest-first` to valid choices |
| **Key code** | `SchedulePolicy.calc_priority()` return sorted by `max_new_tokens - len(output_ids)` |
| **Test** | Send 3 requests with different `max_new_tokens` (50, 200, 100); verify shortest completes first |
| **Expected Result** | Requests with fewer remaining tokens are scheduled before longer ones |

### Exercise C: Add A Prometheus Metric (Medium, ~3 hours)

| Aspect | Detail |
|---|---|
| **Change** | Add `sglang:my_custom_metric` to track something not currently tracked (e.g., average prompt length per batch) |
| **Files** | `metrics_collector.py` — add gauge; `scheduler.py` or `scheduler_metrics_mixin.py` — update gauge in event loop |
| **Key code** | Follow pattern of existing metrics: define in `__init__`, update in event loop, expose via `/metrics` |
| **Test** | Send requests; verify metric appears in `/metrics` with correct values |
| **Expected Result** | New metric visible in Prometheus `/metrics` output |

### Exercise D: Add Config Validation (Easy, ~1 hour)

| Aspect | Detail |
|---|---|
| **Change** | Add validation in `server_args.py` `__post_init__` that warns if `--host 0.0.0.0` is set without `--api-key` |
| **Files** | `server_args.py` — add check in `__post_init__` |
| **Key code** | After arg parsing, check `self.host != "127.0.0.1" and not self.api_key` → emit warning |
| **Test** | Start with `--host 0.0.0.0` without `--api-key` — verify warning appears |
| **Expected Result** | Warning emitted when server is exposed to network without authentication |

## 6. Build A Minimal Demo

### Demo: SGLang-Powered Chatbot (2-3 hours)

```python
"""
Minimal chatbot using SGLang Engine API.
Demonstrates: Engine initialization, generate(), streaming, multi-turn conversation.
"""
import sglang as sgl
from sglang import Engine

# Step 1: Start engine (programmatic, no CLI needed)
engine = Engine(
    model_path="meta-llama/Llama-3.2-3B-Instruct",
    mem_fraction_static=0.85,
    log_level="info",
)

# Step 2: Define a conversation with the frontend DSL
@sgl.function
def chat(s, system_prompt, user_message):
    s += sgl.system(system_prompt)
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen("response", max_tokens=256, temperature=0.7))

# Step 3: Run conversation
system_prompt = "You are a helpful coding assistant. Be concise."
user_message = "Write a Python function to check if a string is a palindrome."

# Using frontend DSL
runtime = sgl.Runtime(model_path="meta-llama/Llama-3.2-3B-Instruct")
runtime.run(chat, system_prompt, user_message)

# Or using raw Engine API
result = engine.generate(
    f"<|system|>{system_prompt}<|user|>{user_message}<|assistant|>",
    sampling_params={"max_new_tokens": 256, "temperature": 0.7},
)
print(result["text"])

# Step 4: Streaming
for chunk in engine.generate(
    "Tell me a short joke.",
    sampling_params={"max_new_tokens": 100},
    stream=True,
):
    print(chunk["text"], end="", flush=True)

# Step 5: Shutdown
engine.shutdown()
```

**What this demo teaches:**
- Programmatic Engine API (no HTTP server needed for single-user)
- Frontend DSL (`sgl.function`, `sgl.gen`, `sgl.system`, `sgl.user`, `sgl.assistant`)
- Raw `engine.generate()` with sampling params
- Streaming responses
- Graceful shutdown

## 7. Contribution Path

### 7.1 Getting Started

| Step | Action | Evidence/Link | Output |
|---|---|---|---|
| 1 | Read contributing guidelines | Check repo for `CONTRIBUTING.md` or similar | Understanding of contribution process |
| 2 | Set up development environment | `pip install -e "python/[all]"` + `pre-commit install` | Working dev environment |
| 3 | Run lint checks locally | `pre-commit run --all-files` | Clean lint pass |
| 4 | Run test suite (minimum) | `cd test && python run_suite.py --stage-a` | Passes on your hardware |
| 5 | Find a `good first issue` | GitHub Issues filtered by `good first issue` label | Claimed issue |
| 6 | Join community | Check README for Slack/Discord/other channels | Community access |

### 7.2 Contribution Types And Difficulty

| Type | Difficulty | Example | Files Typically Changed |
|---|---|---|---|
| **Documentation fix** | Easy | Fix typo, improve docstring, add example | `docs_new/`, README, docstrings |
| **Bug fix (simple)** | Easy-Medium | Fix incorrect error message, edge case in validation | 1-3 files in `srt/` |
| **New model support** | Medium | Add `models/my_model.py` with `EntryClass` | 1 new file + optional config changes |
| **New test** | Medium | Add test for untested feature | `test/srt/test_*.py` |
| **New CLI flag** | Medium | Add `--my-feature` to `ServerArgs` + implement logic | `server_args.py` + 1-3 logic files |
| **New attention backend** | Hard | Implement `AttentionBackend` subclass | `layers/attention/new_backend.py` + registry |
| **New spec algorithm** | Hard | Implement `BaseSpecWorker` subclass | `speculative/new_algo.py` + registry |
| **Performance optimization** | Hard | Improve kernel, reduce overhead in hot path | Varies |
| **New hardware platform** | Very Hard | New `SRTPlatform` + kernel ports | `platforms/` + `hardware_backend/` |

### 7.3 PR Checklist (Inferred From CI)

Based on CI workflows and pre-commit config, a successful PR needs:

1. **Pre-commit passes:** ruff (F401, F821), isort, black, codespell, clang-format (for C++/CUDA)
2. **No large files:** `check-added-large-files` hook
3. **No merge conflicts:** `check-merge-conflict` hook
4. **No private keys:** `detect-private-key` hook
5. **Test suite passes:** PR CI stages A/B/C depending on what changed
6. **`run-ci` label:** Required for CI to run (enforced by pr-gate.yml)
7. **Registered tests:** If adding new test files, must be registered in CI test registry

### 7.4 Review Focus Areas

| Area | Reviewer Will Check | How To Prepare |
|---|---|---|
| **server_args.py** | Is the new flag necessary? Is default sensible? Is it documented? | Add with conservative default, inline docstring |
| **Model files** | Does load_weights cover all HF weight names? Does forward match expected output? | Test against HF transformers reference |
| **Performance** | Does change degrade throughput or latency? | Run benchmark before/after |
| **Tests** | Does change have test coverage? Do existing tests still pass? | Run relevant test suite locally |
| **Memory** | Does change leak memory? Does it increase peak memory? | Monitor with `--enable-metrics` |

## 8. 30 / 60 / 90 Day Plan

### Days 1-30: Foundation — "I can use, run, and debug SGLang"

| Week | Focus | Reading | Building | Deliverable |
|---|---|---|---|---|
| **1** | Orientation | README, pyproject.toml, cli/, launch_server.py, http_server.py, server_args.py (first 500 lines) | Install SGLang, start server with a small model, send requests via curl | Running server, first generated text |
| **2** | Request pipeline | io_struct.py, engine.py, tokenizer_manager.py, scheduler.py (event loops) | Lab 2: trace a request end-to-end with debug logging | Annotated request trace diagram |
| **3** | Batching + caching | schedule_batch.py, radix_cache.py, schedule_policy.py | Lab 3: observe scheduler batching behavior; experiment with scheduling policies | Write-up: "How SGLang batches requests" |
| **4** | GPU execution | model_runner.py, radix_attention.py, attention_registry.py | Lab 7: deploy with Prometheus + Grafana; observe metrics under load | Working monitoring dashboard |

### Days 31-60: Intermediate — "I can modify, extend, and test SGLang"

| Week | Focus | Reading | Building | Deliverable |
|---|---|---|---|---|
| **5** | Model support | model_config.py, model_loader/loader.py, models/registry.py, models/llama.py | Lab 4: add support for a custom model variant | Working custom model |
| **6** | Testing | test/srt/ directory, run_suite.py, CI workflows | Lab 5: write a new integration test for an untested edge case | Submitted test PR |
| **7** | Extension points | plugins/__init__.py, attention_registry.py, spec_registry.py, grammar backends | Exercise B or C: add a scheduling policy or Prometheus metric | Working feature on local branch |
| **8** | Performance | cuda_graph_runner.py, overlap_utils.py, memory_pool.py, flashinfer_backend.py | Lab 6: profile and identify a bottleneck | Profile analysis report |

### Days 61-90: Advanced — "I can contribute significant features"

| Week | Focus | Reading | Building | Deliverable |
|---|---|---|---|---|
| **9** | Advanced features | speculative/ (eagle_worker.py, eagle_info.py, ngram_info.py), disaggregation/ (base/conn.py, mooncake/, mori/) | Implement a small feature from the "Quick Wins" list below | Feature branch with tests |
| **10** | Distributed | distributed/parallel_state.py, layers/moe/token_dispatcher/, eplb/ | Run multi-GPU tests with TP=2 or DP=2 | Multi-GPU working setup |
| **11** | Contribution | .github/workflows/pr-test.yml, CONTRIBUTING guidelines | Submit a real PR: fix a `good first issue` or add a small feature | Merged PR |
| **12** | Production | docker/Dockerfile, examples/monitoring/, environ.py | Deploy SGLang behind nginx with TLS, auth, rate limiting, monitoring | Production deployment runbook |

## 9. Quick Wins

These are concrete, small-scope tasks that build understanding quickly:

| # | Task | Difficulty | Time | Learning Value |
|---|---|---|---|---|
| 1 | Add a `--log-startup-time` flag that prints how long each startup phase took | Easy | 2 hours | Understands startup lifecycle, server_args.py pattern |
| 2 | Add input validation to reject `max_new_tokens=0` requests early | Easy | 1 hour | Understands request validation flow |
| 3 | Write a script that benchmarks TTFT vs batch size and plots results | Easy | 3 hours | Understands batching performance dynamics |
| 4 | Add docstring to an undocumented `ServerArgs` field | Trivial | 15 min | Learns server_args.py conventions |
| 5 | Add `--api-key` warning when `--host 0.0.0.0` without auth | Easy | 30 min | Security awareness, startup validation |
| 6 | Create a minimal Docker Compose with SGLang + nginx reverse proxy | Easy | 2 hours | Production deployment pattern |
| 7 | Add a test for an edge case in sampling (e.g., temperature=0 deterministic output) | Medium | 2 hours | Testing patterns, sampling logic |
| 8 | Write a Python script that uses Engine API for batch inference on a dataset | Easy | 2 hours | Engine API, batching |
| 9 | Convert one `print()` in scheduler to use `logger.debug()` | Trivial | 10 min | Logging conventions |
| 10 | Document one "tribal knowledge" env var from environ.py with usage example | Easy | 30 min | Configuration system understanding |

## 10. Architecture Lessons To Transfer

These are SGLang's reusable architectural patterns that apply beyond this specific project:

### Pattern 1: Subprocess Microkernel With ZMQ IPC

**What:** Three specialized processes (TokenizerManager, Scheduler, DetokenizerManager) communicating via ZMQ PUSH/PULL/DEALER sockets, forked from a parent Engine process.

**Why it works:**
- GIL-isolation: CPU-heavy tokenization doesn't block GPU scheduling
- Fault isolation: one process crashing doesn't take down others (watchdog restarts)
- Language boundary: could replace any component with a different language (Rust gRPC already exists)
- Async/sync boundary: TokenizerManager is asyncio (FastAPI), Scheduler is sync (GPU-bound)

**When to use:** Any serving system where CPU preprocessing and GPU computation have different performance profiles. Similar patterns seen in: vLLM, TGI.

**Evidence:** engine.py:174 `_launch_subprocesses()`, scheduler.py:1537 event loop

### Pattern 2: Mixin-Based Scheduler Composition

**What:** The Scheduler class inherits from 12+ mixin classes, each adding a capability (profiling, metrics, overlap scheduling, output processing, etc.).

**Why it works:**
- Modular: each concern in its own file (~500-2000 lines each instead of one 10,000+ line file)
- Testable: mixins can be tested in isolation
- Extensible: new capabilities added as new mixins

**When to use:** When a core class grows too large (>2000 lines) and has orthogonal concerns. Avoid if mixins have complex interdependencies.

**Evidence:** scheduler.py:324-337 class definition inheriting from 12 mixins

### Pattern 3: Registry + Auto-Discovery

**What:** `ModelRegistry` auto-scans the `models/` directory for `EntryClass`, `AttentionRegistry` uses `@register_attention_backend()` decorator, `SpecRegistry` uses `@register_algorithm()`.

**Why it works:**
- Zero-config: add a file → automatically registered
- Decoupled: registration modules don't import what they register
- Extensible: external packages can register via setuptools entry_points

**When to use:** Any plugin system where you want "add a file and it works" semantics.

**Evidence:** models/registry.py `pkgutil.iter_modules`, attention_registry.py `@register_attention_backend`

### Pattern 4: Overlap Scheduling With Future Values

**What:** The overlap scheduler defers processing batch N's results until batch N+1 starts, using a circular buffer (`FutureMap`) to track unresolved token IDs.

**Why it works:**
- Zero-copy: future indices resolve in-place without copying tensors
- Predictable memory: circular buffer has fixed max size
- ~20-30% throughput improvement over non-overlap mode

**When to use:** Any pipeline where step N+1's preparation can overlap with step N's GPU computation.

**Evidence:** overlap_utils.py `FutureMap` (line 121), scheduler.py `event_loop_overlap` (line 1564)

### Pattern 5: Radix Tree For Prefix Matching

**What:** A trie (radix tree) where each node stores KV cache indices for a token sequence. New requests match the longest common prefix in O(match_length) time.

**Why it works:**
- O(depth) lookup instead of O(N) brute force
- Memory-efficient: shared prefixes share KV cache
- Eviction-friendly: leaf nodes are safe to evict

**When to use:** Any caching scenario with sequential data where prefixes are shared (DNA sequences, file paths, URL routing, log parsing).

**Evidence:** radix_cache.py:360 `match_prefix()`, radix_cache.py:645 `_match_prefix_helper`

### Pattern 6: EnvField Descriptor System

**What:** `EnvField` descriptors on an `Envs` class provide typed, documented, auto-completing access to 80+ environment variables with built-in validation, deprecation, and context-manager-based temporary overrides.

**Why it works:**
- Self-documenting: each env var has a name, type, default, and inline comment
- Safe: `envs.FLAG.get()` pattern prevents accidentally treating descriptors as booleans
- Testable: `envs.FLAG.override(value)` context manager for tests

**When to use:** Any project with many environment variables that needs type safety, documentation, and testability.

**Evidence:** environ.py:38-106 `EnvField` class, environ.py:159-661 80+ `EnvField` instances

### Pattern 7: CUDA Graph Capture For Decode

**What:** Capture the entire decode forward pass (attention + MLP + sampling) into a CUDA graph for each batch size. During serving, replay the graph instead of launching individual kernels.

**Why it works:**
- Eliminates per-kernel launch overhead (100s of microseconds per batch)
- Eliminates Python overhead inside the forward pass
- ~2-5x faster decode compared to eager execution

**When to use:** Any repetitive GPU workload with fixed shapes where the computational graph doesn't change between iterations.

**Evidence:** cuda_graph_runner.py:850 `capture()`, cuda_graph_runner.py:1307 `replay()`

### Key Anti-Pattern To Avoid: God Config Object

SGLang's `server_args.py` (7,950 lines) is a single dataclass with hundreds of fields imported by nearly every module. This creates:

- **Universal dependency:** changing a field requires understanding all consumers
- **Import time bloat:** importing `ServerArgs` imports 15+ srt submodules
- **Test friction:** every test must construct a full `ServerArgs`

**Better approach:** Group config into domain-specific dataclasses (SchedulerConfig, ModelConfig, CacheConfig, etc.) with clear ownership boundaries. SGLang already partially does this with `ModelConfig` and `DeviceConfig` — extend the pattern.

**Evidence:** server_args.py imports from 15+ srt submodules; `ServerArgs` referenced across 100+ files

---

## Appendix: Learning Resource Index

| Resource | Type | URL/Location |
|---|---|---|
| SGLang official docs | Documentation | docs_new/ directory |
| SGLang blog | Technical deep-dives | sgl-project.github.io (inferred from docs) |
| vLLM paper (reference architecture) | Paper | "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023) |
| RadixAttention blog post | Blog | Search "SGLang RadixAttention" (the feature that differentiates SGLang) |
| Continuous batching explainer | Concept | vLLM continuous batching blog post (same mechanism used by SGLang) |
| FlashInfer documentation | Kernel library | flashinfer.ai or GitHub repo |
| NVIDIA CUDA graphs | GPU concept | CUDA Programming Guide, Section 3.2.5 |
| ZMQ guide | IPC | zguide.zeromq.org |
| OpenTelemetry Python | Tracing | opentelemetry-python.readthedocs.io |
| Triton language | GPU kernels | triton-lang.org |
| NCCL documentation | Distributed GPU communication | docs.nvidia.com/deeplearning/nccl |
