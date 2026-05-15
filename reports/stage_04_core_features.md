# Core Features And Call Chains

## 1. Feature Inventory

| Feature | User Value | Entry | Implementation | Tests | Priority |
|---|---|---|---|---|---|
| **Continuous Batching + RadixAttention** | Low-latency, high-throughput serving; automatic prefix reuse | Engine.generate() → Scheduler event loop | scheduler.py, radix_cache.py, schedule_policy.py | test/srt/ | ★★★★★ |
| **Speculative Decoding (Eagle)** | 2-3x faster generation via draft-then-verify | --speculative-algorithm EAGLE | speculative/eagle_worker.py, eagle_utils.py | test/srt/ | ★★★★★ |
| **Prefill-Decode Disaggregation** | Separate prefill/compute scaling; higher throughput | --disaggregation-mode prefill/decode | disaggregation/prefill.py, decode.py, common/conn.py | test/srt/ | ★★★★☆ |
| **Structured/Constrained Output** | JSON/Regex/Grammar-constrained generation | --json-schema, --regex | constrained/grammar_manager.py, xgrammar_backend.py | test/srt/ | ★★★★☆ |
| **Model Forward Pass Pipeline** | Multi-model, multi-architecture inference | ModelRunner.forward() | model_runner.py, layers/radix_attention.py, layers/sampler.py | test/srt/ | ★★★★★ |
| **Distributed Inference (TP/EP/PP)** | Scale to trillion-parameter models | --tp-size, --ep-size, --pp-size | distributed/parallel_state.py, tp_worker.py, moe/ | test/srt/ | ★★★★★ |
| **KV Cache Management** | Memory-efficient long-context serving | RadixCache, MemoryPool | mem_cache/memory_pool.py, radix_cache.py, allocator.py | test/srt/ | ★★★★★ |
| **LoRA Multi-Adapter Serving** | Serve 100s of LoRA adapters simultaneously | --enable-lora, --lora-paths | lora/lora_manager.py, lora/backend/ | test/srt/ | ★★★☆☆ |
| **Quantization (FP8/FP4/INT4/AWQ/GPTQ)** | Serve larger models on less hardware | --quantization fp8 | layers/quantization/, model_loader/ | test/srt/ | ★★★★☆ |
| **Hardware Abstraction (GPU/TPU/MLX/NPU)** | Run on any supported hardware | SGLANG_PLATFORM env var | platforms/, hardware_backend/ | test/srt/ | ★★★☆☆ |

## 2. Top Feature Deep Dives

---

### Feature 1: Continuous Batching with RadixAttention

#### Purpose
The flagship feature of SGLang. Combines **continuous batching** (dynamically add/remove requests from the active batch) with **RadixAttention** (automatic prefix caching via a radix tree). Together they eliminate redundant KV cache computation for shared prompt prefixes and maximize GPU utilization.

#### Entry Points
| Type | Path/Command/API | Evidence |
|---|---|---|
| HTTP API | POST /v1/completions, POST /v1/chat/completions | http_server.py OpenAIServing endpoints |
| HTTP API | POST /generate | http_server.py native endpoint |
| Engine API | engine.generate(prompt, sampling_params) | engine.py generate() method |
| Scheduler | Scheduler.event_loop_normal() | scheduler.py:1537 |

#### Call Chain
```text
HTTP POST /generate
  → http_server.py:generate_request(obj, request)
  → _global_state.tokenizer_manager.generate_request(obj, request)
    → TokenizerManager.generate_request()                           [tokenizer_manager.py:516]
      → _tokenize_one_request()                                      [tokenizer_manager.py]
      → _send_one_request() → ZMQ push to scheduler                  [tokenizer_manager.py:1176]
      → _wait_one_response() → await state.event.wait()             [tokenizer_manager.py:1278]
  ──ZMQ IPC──→
  Scheduler.event_loop_normal()                                      [scheduler.py:1537]
    → recv_requests() → read from ZMQ pull socket                    [scheduler.py]
    → process_input_requests() → create Req objects, add to waiting  [scheduler.py]
    → get_next_batch_to_run()                                        [scheduler.py:2485]
      → SchedulePolicy.get_next_batch()                              [schedule_policy.py]
        → RadixCache.match_prefix() → find shared prefix in KV cache [radix_cache.py]
        → Merge new requests into running batch (if slots available)
        → Evict least recently used cache entries if memory tight    [evict_policy.py]
    → run_batch(batch)                                               [scheduler.py:2996]
      → model_worker.forward_batch_generation(batch)                 [tp_worker.py]
        → ModelRunner.forward()                                      [model_runner.py]
          → ForwardBatch preparation (attention metadata, positions)
          → CudaGraphRunner.replay() or eager forward                [cuda_graph_runner.py]
          → RadixAttention.forward() per layer                       [layers/radix_attention.py]
            → attn_backend.forward_decode() or forward_extend()       [attention backends]
              → Read/write KV cache tensors                          [mem_cache/]
          → LogitsProcessor → LM head                                [layers/logits_processor.py]
          → Sampler.forward() → sample next token                    [layers/sampler.py]
    → process_batch_result(batch, result)                            [scheduler.py]
      → Extract output tokens
      → Update RadixCache with new tokens                            [radix_cache.py:insert()]
      → Send token IDs to DetokenizerManager via ZMQ                 [scheduler.py]
  ──ZMQ IPC──→
  DetokenizerManager:
    → Detokenize token IDs to text
    → Send BatchStrOutput back to TokenizerManager via ZMQ
  ──ZMQ IPC──→
  TokenizerManager.handle_loop()                                     [tokenizer_manager.py:1638]
    → recv from detokenizer socket
    → _handle_batch_output() → update ReqState.out_list
    → set ReqState.event → wake _wait_one_response()
    → Return streaming/non-streaming response to HTTP client
```

#### Key Classes / Functions
| Symbol | File | Responsibility | Evidence |
|---|---|---|---|
| `Scheduler` | managers/scheduler.py:324 | Central orchestrator: batch formation, KV cache, model forward | Inherits 12+ mixins |
| `SchedulePolicy` | managers/schedule_policy.py | Strategy for selecting next batch from waiting queue | FCFS, priority, lpm (longest prefix match) |
| `RadixCache` | mem_cache/radix_cache.py | Radix tree data structure for prefix caching; match_prefix(), insert(), evict() | Based on radix tree algorithm |
| `ScheduleBatch` | managers/schedule_batch.py | Batch of requests: token IDs, positions, attention metadata | Central batch abstraction |
| `Req` | managers/schedule_batch.py | Single request state: input_ids, output_ids, sampling params | Full request lifecycle |
| `ModelRunner` | model_executor/model_runner.py:326 | GPU model execution: forward pass, CUDA graphs, memory pools | Main orchestrator |
| `RadixAttention` | layers/radix_attention.py:53 | Attention layer wrapper: dispatches to backend, reads/writes KV cache | Used by every model |
| `TokenizerManager` | managers/tokenizer_manager.py:216 | Tokenization, request state tracking, IPC to/from scheduler | Main process component |

#### State Changes And Side Effects
1. **Request creation**: `Req` object created with `input_ids`, `sampling_params`, unique `rid`
2. **Batch formation**: Requests merged into `ScheduleBatch`; `RadixCache` consulted for prefix matches
3. **KV cache write**: During forward pass, attention backends write K/V tensors into memory pool
4. **Token generation**: Sampler produces `batch_next_token_ids` (one per request)
5. **Cache update**: `RadixCache.insert()` stores new token sequence for future prefix matches
6. **Request completion**: Req removed from batch when EOS token generated or max_tokens reached
7. **Response dispatch**: Token IDs sent to DetokenizerManager → detokenized → returned to client

#### Error Chain
| Error | Origin | Propagation | Handling | User-visible Result |
|---|---|---|---|---|
| Tokenization error | tokenizer_manager.py `_tokenize_one_request()` | Returned as error in ReqState | Caught, logged | HTTP 400 with error message |
| OOM during forward | model_runner.py `forward()` | CUDA OOM exception | Caught by scheduler, batch retry with smaller size | May cause request abort |
| Invalid sampling params | sampler.py `Sampler.forward()` | Exception raised | Caught by scheduler, logged | HTTP 400 |
| KV cache full | radix_cache.py `evict()` | Eviction policy triggers | LRU/LFU eviction | Potential latency spike |
| Subprocess crash | Scheduler/detokenizer process | ZMQ socket disconnect | Watchdog detects, engine restarts | Service interruption |

#### Test Coverage
| Test File | Scenario | Assertion | Gap |
|---|---|---|---|
| test/srt/test_radix_cache.py | Prefix matching, eviction policies | Correct match, correct eviction order | Large-scale stress test |
| test/srt/test_scheduler.py | Batch formation, priority scheduling | Correct batch content, correct ordering | Edge cases with partial completions |
| test/srt/test_bench_serving.py | End-to-end serving throughput | Performance regression | Varying batch sizes |

#### Modification Opportunities
| Change | Files | Difficulty | Risk |
|---|---|---|---|
| Add new scheduling policy (e.g., shortest-first) | schedule_policy.py | Low | Low (new strategy pattern) |
| Add new eviction policy | mem_cache/evict_policy.py | Low | Low (new strategy) |
| Optimize batch merging logic | scheduler.py `get_next_batch_to_run()` | Medium | Medium (core scheduling) |
| Add custom prefix filter for RadixCache | radix_cache.py | Medium | Low (isolated component) |

---

### Feature 2: Speculative Decoding (Eagle)

#### Purpose
Accelerate autoregressive generation by using a lightweight **draft model** to predict multiple future tokens, then having the **target model** verify all drafts in a single forward pass. Correct drafts are accepted; incorrect ones are discarded. Eagle achieves 2-3x speedup in tokens/second.

#### Entry Points
| Type | Path/Command/API | Evidence |
|---|---|---|
| CLI flag | `--speculative-algorithm EAGLE` | server_args.py |
| Scheduler | `SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)` | scheduler.py:389 |

#### Call Chain
```text
Scheduler runs spec-aware event loop
  → get_next_batch_to_run() → forms draft batch + target batch
  → run_batch()
    → [Target Model Forward] EagleWorker.forward_target()
      → ModelRunner.forward(forward_batch)            [model_runner.py]
        → Standard attention + sampling for target model
    → [Draft Model Forward] EagleWorker.draft()
      → DraftModel forward passes (speculative_num_steps times)
        → build_tree_kernel_efficient()               [eagle_utils.py]
          → Construct tree of candidate tokens via top-k
        → EAGLEDraftCudaGraphRunner.replay()          [cuda_graph_runner]
        → Each step: RadixAttention.forward_decode() for draft model
    → [Verify] EagleWorker.verify()
      → Target model forward with tree attention
        → EagleVerifyInput: carries tree structure (positions, indices)
        → Tree attention: verify all paths in one forward
        → EagleVerifyOutput: accept_indices (which drafts were correct)
    → [Sample] Extract accept_tokens (verified drafts + bonus token)
    → [Extend] EagleWorker.draft_extend()
      → Update draft model KV cache to match accepted tokens
      → EAGLEDraftExtendCudaGraphRunner.replay()
```

#### Key Classes / Functions
| Symbol | File | Responsibility | Evidence |
|---|---|---|---|
| `EagleWorker` | speculative/eagle_worker.py | Orchestrates draft-verify-extend cycle | Eagle v1 worker |
| `EagleWorkerV2` | speculative/eagle_worker_v2.py | Overlap scheduling variant | Pipeline overlap |
| `BaseSpecWorker` | speculative/base_spec_worker.py | Abstract spec worker interface | draft(), verify(), extend() |
| `SpeculativeAlgorithm` | speculative/spec_info.py | Enum for algorithm selection | NONE, EAGLE, NGRAM, etc. |
| `build_tree_kernel_efficient` | speculative/eagle_utils.py | Build draft tree with top-k branching | CUDA kernel |
| `EagleVerifyInput/Output` | speculative/eagle_info.py | Tree structure for verify pass | Positions, retrive indices |
| `SpecRegistry` | speculative/spec_registry.py | Plugin registry for custom algorithms | register_algorithm() |

#### State Changes And Side Effects
1. **Draft token generation**: Draft model produces `num_speculative_steps × top_k` candidate tokens in a tree
2. **Tree attention**: Target model processes all tree paths in a single forward pass
3. **Acceptance check**: Compare draft output logits against target model's predictions
4. **Bonus token**: Target model always produces one additional token (the "+1" bonus)
5. **KV cache sync**: Draft model's KV cache updated to match accepted token positions

#### Error Chain
| Error | Origin | Propagation | Handling | User-visible Result |
|---|---|---|---|---|
| Draft model OOM | eagle_worker.py | CUDA exception | Fallback to non-spec decoding | Performance degradation |
| Tree construction error | eagle_utils.py | Invalid tree structure | Logged, batch retry | Request abort |
| Draft/target model mismatch | eagle_worker.py | Hidden state dimension mismatch | Configuration error → startup failure | Won't start |

#### Modification Opportunities
| Change | Files | Difficulty | Risk |
|---|---|---|---|
| Register custom spec algorithm | spec_registry.py `register_algorithm()` | Low | Low |
| Implement NGram spec (no draft model) | ngram_worker.py (exists) | Reference existing | Low |
| Tune adaptive spec step count | adaptive_spec_params.py | Low | Low |
| Add new draft model architecture | eagle_draft_cuda_graph_runner.py | Medium | Medium |

---

### Feature 3: Structured/Constrained Decoding

#### Purpose
Force LLM output to conform to a specific format: JSON schema, regex pattern, EBNF grammar, or structural tags. Used for API responses, tool calling, and structured data extraction.

#### Entry Points
| Type | Path/Command/API | Evidence |
|---|---|---|
| CLI/API | `--json-schema`, `--regex`, `--grammar-backend` | server_args.py |
| HTTP API | `response_format: {type: json_schema, json_schema: {...}}` | OpenAI compatible endpoint |
| Request-level | `GenerateReqInput.json_schema`, `GenerateReqInput.regex` | managers/io_struct.py |

#### Call Chain
```text
Request with JSON schema arrives
  → TokenizerManager.generate_request()
  → GrammarManager.process_req_with_grammar(req)           [grammar_manager.py]
    → Check for json_schema / regex / ebnf / structural_tag
    → GrammarBackend cache lookup                          [cache: Dict[str, GrammarObject]]
    → Cache HIT: return cached GrammarObject
    → Cache MISS: submit async compilation to ThreadPoolExecutor
      → XGrammarBackend.dispatch_json(json_schema)         [xgrammar_backend.py]
        → CompiledGrammar + GrammarMatcher from xgrammar library
        → Wrap in XGrammarGrammar(BaseGrammarObject)
      → Store in cache, add Future to grammar_queue
  → Scheduler: on batch preparation
    → GrammarManager.get_ready_grammar_requests()
      → Poll grammar_queue for completed Futures
      → all_gather_object() across DP ranks                 [consensus]
      → Attach GrammarObject to Req
  → Scheduler: before sampling
    → GrammarObject.allocate_vocab_mask()                   [bitmask tensor]
    → GrammarObject.fill_vocab_mask(mask, idx)              [allowed tokens]
    → GrammarObject.apply_vocab_mask(logits, mask)          [-inf for disallowed]
  → Sampler → sample from masked logits
  → After sampling:
    → GrammarObject.accept_token(sampled_token_id)          [advance state]
    → GrammarObject.is_terminated() → request complete
```

#### Key Classes / Functions
| Symbol | File | Responsibility | Evidence |
|---|---|---|---|
| `GrammarManager` | constrained/grammar_manager.py | Coordinates grammar lifecycle: submission, polling, DP consensus | Scheduler mixin |
| `BaseGrammarBackend` | constrained/base_grammar_backend.py | Abstract backend: dispatch_json(), cache, ThreadPoolExecutor | Strategy pattern |
| `XGrammarBackend` | constrained/xgrammar_backend.py | XGrammar library backend (Microsoft) | Most full-featured |
| `OutlinesBackend` | constrained/outlines_backend.py | Outlines library backend | Regex+JSON |
| `LlguidanceBackend` | constrained/llguidance_backend.py | llguidance library backend | Guidance-based |
| `BaseGrammarObject` | constrained/base_grammar_backend.py | Per-request grammar: accept_token(), fill_vocab_mask(), rollback() | State machine |
| `ReasonerGrammarBackend` | constrained/reasoner_grammar_backend.py | Wrapper for reasoning models (think/generate phases) | Phase switching |

#### Grammar Backend Comparison
| Backend | Strengths | Weaknesses | Evidence |
|---|---|---|---|
| XGrammar | Fastest, token filtering, reasoning support, jump-forward | XGrammar C++ dependency | xgrammar_backend.py |
| Outlines | Mature, well-tested | Slower, no token filtering | outlines_backend.py |
| llguidance | Guidance-native, EBNF | Newer, less tested | llguidance_backend.py |

#### Modification Opportunities
| Change | Files | Difficulty | Risk |
|---|---|---|---|
| Add new grammar backend | New file in constrained/ + registry | Low | Low |
| Add token filtering rules | xgrammar_backend.py `set_token_filter()` | Low | Low |
| Custom reasoner phase logic | reasoner_grammar_backend.py | Low | Low |

---

### Feature 4: Prefill-Decode Disaggregation

#### Purpose
Split LLM serving into separate **prefill servers** (compute-heavy, process full prompt) and **decode servers** (memory-bound, generate tokens one-by-one). Enables independent scaling of compute and memory, and allows KV cache transfer from prefill to decode nodes.

#### Entry Points
| Type | Path/Command/API | Evidence |
|---|---|---|
| CLI | `--disaggregation-mode prefill` / `--disaggregation-mode decode` | server_args.py |
| Scheduler mixin | `SchedulerDisaggregationPrefillMixin` / `SchedulerDisaggregationDecodeMixin` | scheduler.py:324-337 |

#### Call Chain (Prefill → Decode Transfer)
```text
[Prefill Server]
  → PrefillBootstrapQueue: initialize KVSender per request
  → Scheduler: prefill forward pass (extend mode)
    → ModelRunner.forward() → KV cache populated
  → send_kv_chunk() → transfer KV pages via RDMA
    → CommonKVSender / MooncakeKVSender
    → RDMA write to decode node's GPU memory

[Decode Server]
  → DecodePreallocQueue: initialize KVReceiver per request
  → Pre-allocate KV cache slots (reserve pages)
  → DecodeTransferQueue: poll for transfer completion
    → KVPoll state machine: Bootstrapping → WaitingForInput → Transferring → Success
  → On Success: commit output token, metadata
  → get_new_prebuilt_batch() → construct PrebuiltExtendBatch
    → Merge into running decode batch
  → Standard decode loop
```

#### Transfer Backends
| Backend | File | Method | Use Case |
|---|---|---|---|
| Mooncake | disaggregation/mooncake/conn.py | RDMA via mooncake-transfer-engine | Production (InfiniBand) |
| Mori | disaggregation/mori/conn.py | RDMA via Mori | Alternative RDMA |
| Nixl | disaggregation/nixl/conn.py | RDMA via NIXL | Alternative RDMA |
| Ascend | disaggregation/ascend/conn.py | Ascend NPU transfer | Huawei NPU |
| Fake | disaggregation/fake/conn.py | In-process loopback | Testing/debugging |

#### Key Data Structures
| Symbol | File | Responsibility | Evidence |
|---|---|---|---|
| `BaseKVManager` | disaggregation/base/conn.py | Manages transfer state tables, prefill/decode pairs | Abstract base |
| `CommonKVSender` | disaggregation/common/conn.py | Prefill side: send KV pages via RDMA | Production impl |
| `CommonKVReceiver` | disaggregation/common/conn.py | Decode side: receive KV pages | Production impl |
| `KVPoll` | disaggregation/utils.py | Transfer state enum: Failed→Bootstrapping→Waiting→Transferring→Success | State machine |
| `StagingBuffer` | disaggregation/common/staging_buffer.py | Staging for heterogeneous TP transfers | Optimization |

#### Modification Opportunities
| Change | Files | Difficulty | Risk |
|---|---|---|---|
| Add new transfer backend | New directory in disaggregation/ | Medium | Medium |
| Add metadata field to transfer | io_struct.py + disaggregation/utils.py | Low | Low |

---

### Feature 5: Model Forward Pass Pipeline

#### Purpose
The core inference pipeline: load a model, tokenize input, run the forward pass through all layers (attention → MLP/MoE → residual), compute logits, sample the next token, and update KV cache. Supports 191 model architectures.

#### Entry Points
| Type | Path/Command/API | Evidence |
|---|---|---|
| ModelRunner | ModelRunner.forward() | model_runner.py |
| Individual request | `engine.generate(prompt, sampling_params)` | engine.py |
| Batch forward | `tp_worker.forward_batch_generation()` | tp_worker.py |

#### Call Chain (Detailed Model Forward)
```text
ModelRunner.forward(forward_batch: ForwardBatch)
  → ForwardBatch preparation
    → Compute attention metadata (seq_lens, positions, cu_seqlens)
    → Set forward_mode (EXTEND for prefill, DECODE for generation)
  → CudaGraphRunner.replay(forward_batch)    [if CUDA graph captured]
    OR
  → model.forward(input_ids, positions, forward_batch)   [eager mode]
    → LlamaModel.forward()
      → For each LlamaDecoderLayer:
        → RMSNorm(input)                              [layers/layernorm.py]
        → LlamaAttention.forward()                    [model-specific]
          → QKV projection (QKVParallelLinear)        [layers/linear.py]
          → RoPE (RotaryEmbedding)                    [layers/rotary_embedding/]
          → RadixAttention.forward(q, k, v, forward_batch)  [layers/radix_attention.py]
            → attn_backend.forward_decode(q, k, v)    [attention backends]
              → Compute attention scores
              → Read KV cache (prefix tokens)          [mem_cache/]
              → Write new KV (current token)           [mem_cache/]
              → Compute output
            → o_proj (RowParallelLinear)              [layers/linear.py]
        → Residual connection (add input)
        → LlamaMLP.forward()
          → gate_up_proj (MergedColumnParallelLinear)  [layers/linear.py]
          → SiLU(gate) * up (SiluAndMul)              [layers/activation.py]
          → down_proj (RowParallelLinear)             [layers/linear.py]
          → [For MoE models]: FusedMoE with top-k routing
        → Residual connection
      → RMSNorm(final)
  → lm_head (ParallelLMHead)                          [layers/vocab_parallel_embedding.py]
    → Convert hidden states to vocab logits
  → LogitsProcessor (FP8 quant, temp scaling)         [layers/logits_processor.py]
  → Sampler.forward(logits_output)                    [layers/sampler.py]
    → Temperature scaling
    → Softmax
    → Top-K / Top-P / Min-P filtering
    → Sample → next_token_ids
    → Compute logprobs
  → Return batch_next_token_ids, logprobs
```

#### Key Abstractions
| Symbol | File | Role in Pipeline |
|---|---|---|
| `ModelRunner` | model_executor/model_runner.py | Top-level orchestrator; owns model, sampler, memory pools, CUDA graphs |
| `RadixAttention` | layers/radix_attention.py | Every model's attention layer wraps this; dispatches to backend |
| `LogitsProcessor` | layers/logits_processor.py | Applies LM head, handles FP8 quantization of logits |
| `Sampler` | layers/sampler.py | Sampling strategies (greedy, top-k, top-p, min-p) with flashinfer acceleration |
| `ForwardBatch` | model_executor/forward_batch_info.py | Batch metadata: seq_lens, positions, attention info, mode |
| `CudaGraphRunner` | model_executor/cuda_graph_runner.py | CUDA graph capture and replay for low-overhead decode |
| `Linear` variants | layers/linear.py | Tensor-parallel-aware linear layers (column/row/merged/QKV) |
| `make_layers()` | srt/utils/common.py:669 | Creates all decoder layers with PP support |

---

---

### Feature 6: Distributed Inference (TP/EP/PP)

#### Purpose
Scale LLM inference beyond a single GPU by distributing model weights (Tensor Parallelism), MoE experts (Expert Parallelism), and layers (Pipeline Parallelism) across many GPUs. Enables serving trillion-parameter models like DeepSeek V3 (671B parameters, 256 experts).

#### Core Mechanism: GroupCoordinator
The `GroupCoordinator` (parallel_state.py:199) wraps PyTorch ProcessGroups, providing multiple communication backends (NCCL, Gloo, CustomAllReduce, MSCCLPP) for different parallel dimensions. Groups are nested:
```
GPU 0-7: TP=2, PP=4
┌─────────────────────────────────────────┐
│ TP Group 0: [GPU0, GPU1]                 │ Layers 0-15
│ TP Group 1: [GPU2, GPU3]                │ Layers 0-15 (replicated)
│ TP Group 2: [GPU4, GPU5]                │ Layers 16-31
│ TP Group 3: [GPU6, GPU7]                │ Layers 16-31
└─────────────────────────────────────────┘
PP Group 0: [GPU0, GPU2, GPU4, GPU6]  (all TP ranks 0)
PP Group 1: [GPU1, GPU3, GPU5, GPU7]  (all TP ranks 1)
```

#### Tensor Parallelism (layers/model_parallel.py)
- **ColumnParallel**: Weight split along output dimension
- **RowParallel**: Weight split along input dimension; requires all-reduce
- **QKVParallelLinear**: Specialized for attention QKV projections with GQA/MQA support
- Automatic parallelization via `tensor_parallel()` function that reads `_tp_plan` from modules

#### Expert Parallelism + Token Dispatch
- **StandardDispatcher**: Local indexing when EP=1 (no communication)
- **FlashInferDispatcher**: NVLink-optimized all-to-all for moderate scale
- **DeepEPDispatcher**: For large-scale EP with RDMA (InfiniBand) — supports normal (high-throughput) and low-latency (decode-optimized) modes
- **FuseEPDispatcher**: NPU-specific fused GMM

Dispatch pattern: `dispatch → permute → expert_compute → unpermute → combine`

#### Expert Parallel Load Balancing (EPLB)
`EPLBManager` (eplb/eplb_manager.py) continuously monitors expert utilization and rebalances expert placement:
1. `ExpertDistributionRecorder` tracks per-expert token counts
2. `_check_rebalance_needed()` triggers when utilization drops below threshold
3. `ExpertLocationMetadata` computes new physical-to-logical mapping
4. Weights hot-swapped across GPUs in chunks (no inference interruption)

#### Pipeline Parallelism
Non-terminal PP stages forward intermediate hidden states (`PPProxyTensors`) to the next stage. Only the last PP stage computes logits and samples. Uses 1F1B-style scheduling with interleaved rank indexing.

#### Evidence
| Component | File | Key Insight |
|---|---|---|
| GroupCoordinator | distributed/parallel_state.py:199 | Multi-backend, multi-group architecture |
| TP plan | layers/model_parallel.py | Colwise/Rowwise automatic parallelization |
| DeepEP dispatcher | layers/moe/token_dispatcher/deepep.py | RDMA + NVLink all-to-all |
| EPLB Manager | eplb/eplb_manager.py | Runtime load-aware expert redistribution |
| PP execution | managers/tp_worker.py:528-539 | Inter-stage hidden state passing |

---

## 3. Cross-feature Patterns

### Pattern 1: Strategy/Backend Registry
Used by: Attention backends, MoE backends, Grammar backends, LoRA backends, Transfer backends, Spec algorithms
```python
# Common pattern:
REGISTRY = {}
def register(name):
    def decorator(fn):
        REGISTRY[name] = fn
        return fn
    return decorator
```
Evidence: attention_registry.py:23-29, lora_registry.py, spec_registry.py, grammar backends

### Pattern 2: Mixin-based Scheduler Composition
The Scheduler composes 12+ capabilities via Python multiple inheritance, each mixin adding specific methods without modifying the core class.
Evidence: scheduler.py:324-337

### Pattern 3: ZMQ IPC Pipeline
All inter-process communication uses ZMQ message passing — no shared memory for request data.
Evidence: engine.py ZMQ imports, io_struct.py message definitions

### Pattern 4: Factory Method Selection
Model loading, attention backend selection, sampler creation, and hardware platform selection all use factory methods that dispatch based on configuration.
Evidence: model_loader/__init__.py `get_model_loader()`, attention_registry.py, sampler.py `create_sampler()`

## 4. Feature Quality Verdict

### Strengths
1. **Deep integration**: Features compose cleanly (spec + constrained, disagg + spec, LoRA + quant)
2. **Pluggability**: Strategy pattern used consistently across backends
3. **Performance-first**: CUDA graphs, overlap scheduling, RDMA transfers are production-grade optimizations
4. **Hardware abstraction**: Platform layer enables NVIDIA/AMD/Intel/TPU/NPU support

### Weaknesses
1. **Configuration complexity**: server_args.py at 7,950 lines reflects the combinatorial explosion of feature flags
2. **Feature interaction testing**: With 10+ composable features, interaction bugs are possible (e.g., spec + disagg + constrained)
3. **Debugging difficulty**: Multi-process + multi-GPU + async makes debugging challenging
4. **Some features are less mature**: DFlash, STANDALONE spec, Nixl transfer appear less tested than Eagle/Mooncake

### Minimum Modification Experiment
To verify understanding of the system, add a simple feature: **`--log-every-request` flag** that logs request ID, prompt length, output length, and latency for every completed request.
- **Files to modify**: server_args.py (add flag), tokenizer_manager.py (add logging in `_handle_batch_output`), scheduler.py (add timing in `process_batch_result`)
- **Difficulty**: Low
- **Files touched**: ~3
- **Test**: Send a request and verify the log line appears
