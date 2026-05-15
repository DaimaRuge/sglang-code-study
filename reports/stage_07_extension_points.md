# Extension Points And Secondary Development

## 1. Extension Point Inventory

| Extension Point | Type | Path/Symbol | How To Use | Stability | Evidence |
|---|---|---|---|---|---|
| **Hardware Platform** | Plugin (entry_point) | plugins/__init__.py `sglang.srt.platforms` | Implement `SRTPlatform` subclass; register via setuptools entry_points | Stable | plugins/__init__.py:28 |
| **General Plugin** | Plugin (entry_point) | plugins/__init__.py `sglang.srt.plugins` | Callable with side effects (register hooks, replace classes) | Stable | plugins/__init__.py:29 |
| **Attention Backend** | Registry | layers/attention/attention_registry.py `register_attention_backend()` | Implement `AttentionBackend` subclass; register with decorator | Internal but stable | attention_registry.py:23-29 |
| **Model Architecture** | Auto-registration | models/registry.py `ModelRegistry` | Create .py file with `EntryClass = MyModelClass` | Stable | registry.py auto-scan |
| **Grammar Backend** | Registry | constrained/base_grammar_backend.py `create_grammar_backend()` | Implement `BaseGrammarBackend` subclass; update factory | Internal | grammar_manager.py |
| **Spec Algorithm** | Registry | speculative/spec_registry.py `register_algorithm()` | Implement `BaseSpecWorker` subclass; register with decorator | Internal | spec_registry.py |
| **LoRA Backend** | Registry | lora/backend/lora_registry.py | Implement `BaseLoRABackend` subclass; register in dict | Internal | lora_registry.py |
| **Transfer Backend** | Directory | disaggregation/base/conn.py | Implement `BaseKVManager`/`KVSender`/`KVReceiver` | Internal | Disaggregation conn.py |
| **Scheduling Policy** | Strategy | managers/schedule_policy.py `SchedulePolicy` | Add policy method and register in policy selection | Internal | schedule_policy.py |
| **MoE Runner** | Strategy | layers/moe/moe_runner/ | Implement new runner backend + dispatcher | Internal | moe/__init__.py |
| **Quantization** | Strategy | layers/quantization/ | Add quantization class; register in QUANTIZATION_METHODS | Internal | quantization/__init__.py |
| **Sampling Strategy** | Strategy | layers/sampler.py `create_sampler()` | Modify sampler backends (flashinfer, pytorch, ascend) | Internal | sampler.py |
| **Tokenizer** | Factory | tokenizer/ `get_tokenizer()` | Custom tokenizer wrapper | Internal | tokenizer/ |

## 2. Plugin / Provider / Adapter / Connector Mechanisms

### 2.1 Hardware Platform Plugin (Formal, Stable)
```
Step 1: Implement SRTPlatform subclass
  class MyPlatform(SRTPlatform):
      def get_default_attention_backend(self):
          return "my_backend"
      def apply_server_args_defaults(self, server_args):
          server_args.attention_backend = "my_backend"

Step 2: Register via setuptools entry_points
  [project.entry-points."sglang.srt.platforms"]
  my_platform = "my_package:create_my_platform"

Step 3: Activate
  export SGLANG_PLATFORM=my_platform
  sglang serve --model-path ...
```
Evidence: plugins/__init__.py:28, platforms/interface.py

### 2.2 Hook System (Formal, Stable)
```python
# Plugins register hooks that inject into SGLang's runtime
from sglang.srt.plugins.hook_registry import HookRegistry

def my_hook(*args, **kwargs):
    # Run before/after the target function
    pass

HookRegistry.register_hook(
    target="sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    hook=my_hook,
    position="after"
)
```
Evidence: plugins/hook_registry.py

### 2.3 Attention Backend (Informal, Stable)
```python
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.attention_registry import register_attention_backend

class MyAttentionBackend(AttentionBackend):
    def init_forward_metadata(self, forward_batch):
        ...
    def forward_decode(self, q, k, v, layer):
        ...
    def forward_extend(self, q, k, v, layer):
        ...

@register_attention_backend("my_custom_backend")
def create_my_backend(runner):
    return MyAttentionBackend(runner)
```
Evidence: attention_registry.py pattern

### 2.4 Model Auto-Registration (Semi-formal, Stable)
```python
# models/my_new_model.py
from sglang.srt.layers.radix_attention import RadixAttention
# ... standard model pattern ...

class MyNewForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None):
        ...
    def load_weights(self, weights_iterator):
        ...

EntryClass = MyNewForCausalLM  # ← Auto-discovered by ModelRegistry
```
Evidence: models/registry.py

### 2.5 Spec Algorithm (Internal, Evolving)
```python
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_registry import register_algorithm

@register_algorithm("MY_SPEC")
class MySpecWorker(BaseSpecWorker):
    def draft(self, batch):
        ...
    def verify(self, batch, draft_output):
        ...
```
Evidence: spec_registry.py

## 3. Hook / Middleware / Event / Registry Mechanisms

| Mechanism | Type | How It Works | Use Case | Evidence |
|---|---|---|---|---|
| **HookRegistry** | Before/After hooks | Register callbacks on any function; applied at plugin load time | Inject logging, modify scheduler behavior | plugins/hook_registry.py |
| **Attention Registry** | Decorator-based factory | `@register_attention_backend("name")` lazily imports backend | Swap attention implementation | attention_registry.py |
| **Model Registry** | Auto-discovery | `pkgutil.iter_modules` scans models/ directory | Add new model architecture | models/registry.py |
| **Plugin entry_points** | setuptools | `entry_points()` discovers installed plugins | Add hardware platform or general feature | plugins/__init__.py |
| **dispatch_event_loop** | Strategy selection | Scheduler selects event loop based on disaggregation mode + overlap | Custom scheduling loop | scheduler.py |
| **ServerArgs extension** | Dataclass fields | New flags added to ServerArgs; plugins can add custom fields | Add configuration options | server_args.py |
| **ZMQ IPC** | Message passing | Custom message types in io_struct.py; Scheduler routes by type | New IPC command | io_struct.py |

## 4. Add A New Feature Path

| Step | Files | Change | Test | Risk |
|---|---|---|---|---|
| 1. Define config | server_args.py | Add `--my-feature` flag to ServerArgs | Validation test | Low |
| 2. Define IPC types | managers/io_struct.py | Add MyFeatureReqInput/Output | Serialization test | Low |
| 3. Add Engine API | engine.py | Add `engine.my_feature()` method | Integration test | Low |
| 4. Add Scheduler logic | managers/scheduler.py | Add handling in event loop or new mixin | Unit test | Medium |
| 5. Add HTTP endpoint | entrypoints/http_server.py | Add route + handler | HTTP test | Low |
| 6. Update docs | docs_new/ | Document usage | Manual review | Low |

### Example: Adding a "Log Every Token" Feature
```
1. server_args.py: add --log-every-token flag (bool, default False)
2. scheduler.py: in process_batch_result(), if flag is True, log token IDs
3. http_server.py: no change needed (token data already in responses)
4. Test: send request, verify log output contains token IDs
Difficulty: Low (~20 lines of code, 3 files)
```

## 5. Add A New Integration Path

| Step | Files | Purpose | Difficulty |
|---|---|---|---|
| 1. **New database backend for caching** | mem_cache/storage/my_db.py | Implement KV storage connector | Medium |
| 2. **New model hub connector** | connector/my_hub.py | Add ModelScope-like remote loader | Medium |
| 3. **New monitoring backend** | observability/my_mon.py | Add custom metrics exporter | Low |
| 4. **New auth provider** | entrypoints/http_server.py | Add API key / OAuth middleware | Low |
| 5. **New tokenizer backend** | tokenizer/my_tokenizer.py | Support non-HF tokenizers | Medium |

## 6. Replace A Dependency Path

| Target | Current | Replacement | Files Affected | Difficulty |
|---|---|---|---|---|
| **HTTP Server** | FastAPI + uvicorn | Granian (already supported) | http_server.py (already flexible) | Low |
| **Attention Backend** | flashinfer | Custom kernels | attention_registry.py + new backend file | Medium |
| **IPC (ZMQ)** | pyzmq | Redis pub/sub | engine.py, managers/* (all IPC init) | High |
| **Model Loader** | DefaultModelLoader | Custom loader | model_loader/loader.py new subclass | Medium |
| **PyTorch** | torch 2.11.0 | MLX (Apple) / JAX (TPU) | hardware_backend/mlx/ + platforms/ | Very High |
| **Tokenizer** | HuggingFace tokenizers | tiktoken only | tokenizer/ | Low |
| **Grammar** | xgrammar | outlines only | constrained/ | Low |

## 7. Business Adaptation Options

| Scenario | Feasibility | Required Changes | Cost | Risk |
|---|---|---|---|---|
| **Use as-is** (serve models via API) | High | None — install + start | Low (ops) | Low |
| **Custom model serving** | High | Add model file in models/ + register | 1-3 days | Low |
| **Private cloud deployment** | High | Docker/k8s config + network security | 1-2 weeks | Medium (GPU infra) |
| **Add authentication layer** | High | Add FastAPI middleware or API gateway | 1-3 days | Low |
| **Custom attention kernel** | Medium | New attention backend + register | 1-3 weeks | Medium |
| **Custom hardware support** | Medium | New platform plugin + kernel porting | 2-8 weeks | Medium-High |
| **Embed in product (OEM)** | Medium | Wrap Engine API + custom config + license review | 2-4 weeks | Low (Apache 2.0) |
| **Fork and add major feature** | Medium | Feature changes + keep up with upstream | Ongoing | High (divergence) |
| **Replace PyTorch backend** | Very Low | Massive refactor across all layers/models | 6+ months | Very High |
| **Use as RL training backbone** | High | Existing integration with verl, AReaL, slime | 0 days | Low |

## 8. Secondary Development Task Ladder

| Level | Task | Files | Duration | Validation | Risk |
|---|---|---|---|---|---|
| **L1: Quick Win** | Add `--log-every-request` flag | server_args.py + tokenizer_manager.py | 1 day | Send request, check logs | Low |
| **L1: Quick Win** | Add new scheduling policy (shortest-first) | schedule_policy.py | 1-2 days | Run scheduler unit test | Low |
| **L1: Quick Win** | Add custom health check endpoint | http_server.py | 1 day | curl /my-health | Low |
| **L2: Medium** | Integrate new model architecture | models/my_model.py | 1-3 days | Run model through serve + test | Low |
| **L2: Medium** | Add custom attention backend | layers/attention/my_backend.py | 3-5 days | Benchmark vs flashinfer | Medium |
| **L2: Medium** | Add custom grammar backend | constrained/my_grammar.py | 3-5 days | Test with JSON schema | Low |
| **L2: Medium** | Add REST API rate limiter | http_server.py middleware | 2-3 days | Send concurrent requests | Low |
| **L3: Deep** | Add custom speculative algorithm | speculative/my_spec.py | 2-3 weeks | Benchmark acceptance rate | High |
| **L3: Deep** | Add new hardware platform support | platforms/ + hardware_backend/ | 4-8 weeks | Run full test suite | High |
| **L3: Deep** | Replace IPC with Redis | engine.py, managers/* | 2-4 weeks | Multi-node test | High |
| **L3: Deep** | Add request prioritization dashboard | http_server.py + frontend | 2-4 weeks | Manual QA | Medium |

## 9. Productionization Gap For Business Use

| Gap | Status | Action Required |
|---|---|---|
| **Authentication** | Basic (`--api-key` only) | Add OAuth/OIDC support or use API gateway |
| **Authorization** | None (all endpoints accessible with key) | Add RBAC or route-level permissions |
| **Rate Limiting** | None built-in | Add at application level or API gateway |
| **TLS/HTTPS** | Not built-in (defer to reverse proxy) | Configure TLS termination at load balancer |
| **Audit Logging** | Basic (prometheus metrics) | Add request-level audit log |
| **Disaster Recovery** | None | Implement model artifact backup, KV cache persistence |
| **Multi-tenancy** | Partial (LoRA isolation, extra_key namespaces) | Add tenant-level quotas and billing |
| **SLA Monitoring** | Limited (Prometheus metrics) | Add SLO tracking, alerting rules |
| **Compliance (SOC2, HIPAA)** | Not addressed | Add data encryption, access controls, audit trails |
| **Capacity Planning** | Manual | Auto-scaling based on queue depth, latency metrics |

## 10. Recommendation

### For Learning
**Excellent** — Well-structured layered architecture, clean plugin patterns, comprehensive test suite. A model for high-performance Python serving systems.

### For Integration
**Very Good** — OpenAI-compatible API, programmatic Engine API, LoRA multi-adapter support. Use as a drop-in replacement for commercial LLM APIs.

### For Secondary Development
**Good** — Pluggable backends (attention, grammar, MoE, LoRA, transfer), auto-registration for models, clean extension points. Main friction: 7,950-line server_args.py and mixin-heavy Scheduler.

### For Production Deployment
**Good, with caveats** — Production-grade performance (RadixAttention, CUDA graphs, overlap scheduling) used at scale. Missing: built-in auth, rate limiting, audit logging — fill with API gateway or custom middleware.

### For Commercialization
**Feasible** — Apache 2.0 license enables embedding/commercialization. Value-add paths: custom hardware support, enterprise security features, managed cloud service. Risk: fast-moving upstream; need strategy to track or contribute back.
