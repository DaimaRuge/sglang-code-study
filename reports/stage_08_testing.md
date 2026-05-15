# Testing, Quality And Debugging

## 1. Test Framework And Layout

| Type | Tool | Directory | Naming | Evidence |
|---|---|---|---|---|
| **Unit/Integration** | pytest | test/srt/ | test_*.py | test/srt/ directory |
| **Parameterized** | parameterized (pytest plugin) | test/srt/ | @parameterized.expand | pyproject.toml test deps |
| **Coverage** | pytest-cov + diff-cover | .coveragerc | .coveragerc config | pyproject.toml test deps |
| **Expect Tests** | expecttest | test/srt/ | expect tests for output matching | pyproject.toml test deps |
| **E2E LLM Eval** | lm-eval | test/lm_eval_configs/ | lm-eval[api] configs | pyproject.toml test deps |
| **Manual Tests** | Custom scripts | test/manual/ | Manual test scripts | test/manual/ directory |
| **Registered Tests** | Custom | test/registered/ | Pre-registered test cases | test/registered/ directory |
| **Suite Runner** | run_suite.py | test/run_suite.py | Orchestrates test execution | test/run_suite.py |
| **Lint** | ruff, isort, codespell | .pre-commit-config.yaml | Pre-commit hooks | .pre-commit-config.yaml |
| **CI** | GitHub Actions | .github/workflows/ | 50+ workflow files | .github/workflows/ directory |

## 2. Test Command Matrix

| Command | Purpose | Source | Result | Notes |
|---|---|---|---|---|
| `cd test && python run_suite.py` | Run full test suite | README, CI workflows | Not executed (no GPU) | Orchestrates multiple test types |
| `pytest test/srt/` | Run unit/integration tests | CI configs | Not executed | - |
| `pytest --cov=sglang test/srt/` | Run with coverage | .coveragerc | Not executed | Requires GPU |
| `pre-commit run --all-files` | Run all lint checks | .pre-commit-config.yaml | Not executed | Should work without GPU |
| `python -m sglang.check_env` | Verify environment | check_env.py | Not executed | Verifies CUDA, PyTorch, GPU |
| CI: pr-test-stage | Per-PR test stage | .github/workflows/_pr-test-stage.yml | CI-passing (inferred from active CI) | Runs on GitHub runners |
| CI: nightly-test-amd | Nightly AMD GPU tests | .github/workflows/nightly-test-amd.yml | CI-passing (inferred) | AMD ROCm |
| CI: lint | Lint check | .github/workflows/lint.yml | CI-passing (inferred) | ruff, isort |

## 3. Unit / Integration / E2E Ratio

| Test Type | Estimated Proportion | Evidence | Confidence |
|---|---|---|---|
| **Unit tests** | ~30% | Individual layer/utility tests in test/srt/ | Medium (from directory scan) |
| **Integration tests** | ~50% | Tests that launch scheduler, model runner, or full engine | Medium |
| **E2E tests** | ~15% | lm-eval based, full request-response cycle | Medium |
| **Manual tests** | ~5% | test/manual/ scripts | Low |

### Coverage Tool Configuration
```ini
# .coveragerc
[run]
source = sglang
omit = test/*, */test_*.py

[report]
exclude_lines = pragma: no cover, if TYPE_CHECKING, raise NotImplementedError
```
Evidence: .coveragerc

## 4. Mock / Fixture / Test Data Patterns

| Pattern | Example | Evidence |
|---|---|---|
| **Dummy Model Loader** | `DummyModelLoader` for testing without real weights | model_loader/loader.py |
| **Fake Transfer Backend** | `disaggregation/fake/conn.py` for disaggregation testing | disaggregation/fake/ |
| **Fixture-based test setup** | pytest fixtures for engine, scheduler, model_runner | test/srt/ fixtures |
| **Parameterized test cases** | `@parameterized.expand` for testing multiple model configs | test/srt/ decorators |
| **CI weight validation** | `model_loader/ci_weight_validation.py` for CI-specific checks | model_loader/ |

## 5. Coverage Blind Spots

| Area | Missing Test | Risk | Suggested Test |
|---|---|---|---|
| **Multi-node distributed** | Tests run on single GPU/GitHub runner | High — distributed bugs only in production | Add multi-GPU CI with model parallelism tests |
| **Large-scale EP (DeepEP)** | No automated test for >8 GPU expert parallelism | High — core production feature | Run on multi-node CI |
| **Error recovery paths** | Limited tests for OOM, subprocess crash recovery | Medium — production resilience | Fault injection tests (kill scheduler, fill memory) |
| **Performance regression** | No automated benchmark comparison | Medium — performance may silently degrade | Add benchmark CI pipeline with thresholds |
| **Multi-modal input validation** | Edge cases with corrupted images, weird aspect ratios | Medium | Add fuzzing for image/video inputs |
| **Streaming response correctness** | Tests mostly check non-streaming | Medium | Add streaming response ordering/corruption tests |
| **Security edge cases** | No penetration tests, input fuzzing | Medium | Add prompt injection, malformed JSON tests |
| **Long-running stability** | No soak tests (hours/days of continuous serving) | Low-Medium | Add 24h continuous serving test |
| **LoRA interaction with spec decoding** | No explicit test for this combination | Low — rare config | Add combination test |

## 6. How To Add A Test

```text
target behavior
→ identify test file (test/srt/test_<feature>.py)
→ set up fixture (engine, model_runner, or mock)
→ write assertion (expected output, metrics, or absence of errors)
→ run: pytest test/srt/test_<feature>.py -xvs
```

### Example: Testing a New Sampling Policy
```python
# test/srt/test_schedule_policy.py
import pytest
from sglang.srt.managers.schedule_policy import SchedulePolicy

def test_shortest_first_policy():
    policy = SchedulePolicy("shortest_first")
    reqs = [
        MockReq(input_len=100),
        MockReq(input_len=50),
        MockReq(input_len=200),
    ]
    sorted_reqs = policy.calc_priority(reqs)
    assert sorted_reqs[0].input_len == 50  # Shortest first
```

## 7. Debugging Guide

| Scenario | Entry | Logs/Tools | Likely Cause | Evidence |
|---|---|---|---|---|
| **Server won't start** | `sglang serve` stderr | Check CUDA version, GPU memory, port availability | Missing CUDA, OOM, port conflict | check_env.py |
| **Request timeout** | TokenizerManager logs | `--log-level debug` | Queue backlog, old request stuck | scheduler.py recv_requests |
| **NaN logits / bad output** | ModelRunner forward | Enable model dump (`--dump-tensor-path`) | Bad model config, FP8 overflow | model_runner.py |
| **OOM during serving** | Scheduler logs | `--mem-fraction-static` too high | KV cache overallocation | scheduler.py memory pool |
| **Slow performance** | Profiler | `POST /start_profile`, `py-spy` | Wrong attention backend, no CUDA graph | profiler.py |
| **Grammar not working** | GrammarManager logs | grammar_queue status, Future errors | Invalid JSON schema, compilation failure | grammar_manager.py |
| **ZMQ connection lost** | Engine logs | Subprocess watchdog log | Scheduler/Detokenizer crash | watchdog.py |
| **LoRA not loading** | LoRAManager logs | `--log-level debug`, check adapter format | Wrong weight names, OOM | lora_manager.py |

### Debug Tools Available
| Tool | How to Use | Purpose |
|---|---|---|
| **py-spy** | `py-spy dump --pid <PID>` | Live Python stack dump (no instrumentation) |
| **PyTorch Profiler** | `POST /start_profile` + `POST /stop_profile` | GPU kernel timing, CPU/GPU trace |
| **Tensor Dump** | `--dump-tensor-path /tmp/dump` | Save intermediate tensors for inspection |
| **Health Check** | `GET /health_generate` | Run real inference for liveness verification |
| **Prometheus Metrics** | `GET /metrics` | Runtime metrics (queue depth, latency, throughput) |
| **Log Levels** | `--log-level debug` | Verbose logging for debugging |

## 8. Code Quality Scorecard

| Dimension | Rating | Evidence | Recommendation |
|---|---|---|---|
| **Linting** | A | .pre-commit-config.yaml (ruff, isort, codespell) | Already excellent |
| **Type Safety** | B- | Some type hints present, not enforced (no mypy CI) | Add mypy to CI with gradual strictness |
| **Code Duplication** | B | 191 model files share patterns but not copy-paste; some layer code similar to vLLM | Continue model abstraction consolidation |
| **Documentation** | B+ | Comprehensive docs, API docs, guides | Add architecture decision records (ADRs) |
| **Error Handling** | B | Good try/except in critical paths; some bare exceptions in older code | Audit bare except clauses |
| **Test Coverage** | B | Good coverage of core features; integration test gaps | Add multi-node and soak tests |
| **Dependency Health** | B+ | Pinned versions, regular updates, automated bump bots | Monitor for CVE in torch, flashinfer |
| **Build Reproducibility** | B | Multiple pyproject variants, complex CUDA deps | Simplify build matrix |

### Overall Quality Grade: B+ (Strong with improvement areas)

## 9. Maintainability Verdict

### Strengths
1. **Automated linting and formatting** — consistent code style
2. **Active CI** — 50+ workflows for PR testing, nightly, platform-specific
3. **Auto-version bumps** — bots for flashinfer, kernel versions
4. **Registry patterns** — adding new models, backends is straightforward
5. **Well-commented core files** — engine.py, scheduler.py have explanatory docstrings

### Weaknesses
1. **server_args.py (7,950 lines)** — single largest maintenance burden
2. **No mypy strict mode** — type errors can slip through
3. **Test coverage gaps** — multi-node, error recovery, performance regression
4. **No automated vulnerability scanning** — dependency CVE checking not in CI
5. **Debugging complexity** — multi-process, multi-GPU, async makes debugging hard

### Maintainability Score: B (Maintainable but requires discipline)
