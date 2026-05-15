# Security, License And Supply Chain

## 1. License Summary

| Item | Value | Evidence | Commercial Impact |
|---|---|---|---|
| **License** | Apache License 2.0 | LICENSE:1-3 | Permissive — commercial use allowed |
| **Copyright** | 2023-2024 SGLang Team | LICENSE header in all source files | Standard Apache 2.0 attribution |
| **Patent Grant** | Yes (Section 3) | Apache 2.0 text | Patent retaliation protection included |
| **Copyleft** | None | Apache 2.0 | No obligation to release derivative source |
| **Attribution** | Required (NOTICE file) | Apache 2.0 Section 4d | Must include NOTICE in distributions |
| **Trademark** | Not granted | Apache 2.0 | Cannot use "SGLang" name for commercial products |
| **Host Organization** | LMSYS (non-profit) | README:81 | Long-term steward commitment |
| **PyPI License** | Apache 2.0 | pyproject.toml:11 | Matches LICENSE file |

### License Compatibility Assessment
- **Commercial use**: ✅ Allowed (explicit grant in Section 2)
- **Distribution**: ✅ Allowed (with NOTICE preservation)
- **Modification**: ✅ Allowed (no copyleft trigger)
- **Patent litigation**: ✅ Protection (retaliation clause)
- **GPL compatibility**: ⚠️ Apache 2.0 is compatible with GPLv3 only
- **Embedding in proprietary products**: ✅ Allowed

## 2. License Compatibility And Obligations

### Third-party Dependency License Risks

| Dependency | License | Risk | Evidence |
|---|---|---|---|
| torch (PyTorch) | BSD-3 | Low | Standard permissive |
| flashinfer | Apache 2.0 | Low | Same license |
| transformers (HuggingFace) | Apache 2.0 | Low | Same license |
| xgrammar | Apache 2.0 | Low | Same license |
| outlines | Apache 2.0 | Low | Same license |
| flash-attn (flash-attn-4) | BSD | Low | Permissive |
| nvidia-cutlass-dsl | NVIDIA license | ⚠️ Medium | Check commercial distribution terms |
| sglang-kernel | Apache 2.0 | Low | Same org |

### Obligations Checklist
- [x] Include NOTICE file in distributions
- [x] Retain copyright headers in source files
- [x] State changes when modifying Apache 2.0 code
- [x] No copyleft obligations (no GPL code in core)
- [ ] Verify NVIDIA CUTLASS DSL license for commercial redistribution

## 3. Dependency And Supply Chain Inventory

| Dependency/Asset | Source | Version | Role | Risk | Evidence |
|---|---|---|---|---|---|
| **PyTorch** | pypi | 2.11.0 | DL framework | Medium (version locked) | pyproject.toml:70 |
| **flashinfer** | pypi | 0.6.11 | CUDA attention kernels | Medium (kernel library) | pyproject.toml:32-33 |
| **transformers** | pypi | 5.6.0 | Model loading | Low | pyproject.toml:78 |
| **sglang-kernel** | pypi | 0.4.2 | Custom CUDA kernels | Low (same org) | pyproject.toml:62 |
| **xgrammar** | pypi | 0.2.0 | Grammar backend | Low | pyproject.toml:82 |
| **fastapi** | pypi | latest | HTTP framework | Low | pyproject.toml:29 |
| **zmq** | pypi | >=25.1.2 | IPC | Low | pyproject.toml:55 |
| **sgl-deep-gemm** | pypi | 0.1.0 | Deep GEMM kernels | Low (same org) | pyproject.toml:63 |
| **ray** | pypi | >=2.54.0 | Distributed (optional) | Low | pyproject.toml:117 |
| **granian** | pypi | >=2.6.0 | HTTP/2 server (optional) | Low | pyproject.toml:129 |
| **torchao** | pypi | 0.17.0 | Quantization | Low | pyproject.toml:71 |
| **Docker images** | GitHub Container Registry | Nightly + release | Container deployment | Low (internal) | .github/workflows/_docker-build-and-publish.yml |
| **GitHub Actions** | GitHub Marketplace | Various | CI/CD | Low | .github/workflows/ |
| **Rust toolchain** | rustup | Nightly/stable | gRPC server build | Low | rust/sglang-grpc/Cargo.toml |

## 4. Vulnerability Scanning Result

| Tool/Method | Result | Limitations |
|---|---|---|
| **Manual dependency review** | Version pins are recent and actively maintained | No CVE scan tool available in this environment |
| **GitHub Dependabot** (inferred from CI) | Likely active for dependency updates | Not verified (no network access) |
| **Version bump bots** | Bots exist for flashinfer, kernel, sglang versions | bot-bump-*.yml workflows |

### Recommendations for Scanning
```bash
# If scanning tools are available:
pip-audit                        # Python dependency CVE scanner
safety check --file requirements.txt  # Safety DB scanner
trivy fs /path/to/sglang         # Container/Docker scanner
```

## 5. Secret And Configuration Security

| Risk | Evidence | Impact | Mitigation |
|---|---|---|---|
| **No hardcoded secrets found** | Source scan — no API keys, tokens in source | Low | Continue best practices |
| **.env.example pattern** | Not found; env vars documented in environ.py | Low | Document env var template |
| **CI secrets** | GitHub Actions secrets (inferred, not inspectable) | Medium | Ensure secret rotation policy |
| **Logging may leak tokens** | DEBUG-level logging could log request contents | Low | Production should use INFO+ level |
| **HTTP API key in CLI** | `--api-key` flag visible in process list | Low | Use env var or file-based key |
| **ZMQ IPC unencrypted** | Local IPC, no transport security | Low (local only) | Bind ZMQ to localhost only |
| **Model weights in plaintext** | No weight-level encryption | Medium | Use encrypted filesystem if required |

## 6. Application Security Review

| Risk Class | Evidence | Impact | Severity | Mitigation |
|---|---|---|---|---|
| **No built-in authentication** | http_server.py: routes have no auth middleware | Unauthorized access to all endpoints | **High** | Set `--api-key` or use API gateway |
| **No rate limiting** | No middleware for request throttling | DoS via request flooding | **Medium** | Add rate limiter or use API gateway |
| **File path traversal** | Model path loading from user input (`--model-path`) | Model loading from arbitrary paths | **Low** (server-side only) | Validate paths, restrict to known directories |
| **Remote code execution** | `transformers` `trust_remote_code=True` for custom models | Arbitrary Python execution from model code | **Medium** (opt-in) | Disable `trust_remote_code` for untrusted models |
| **Input validation** | io_struct.py `normalize_batch_and_arguments()` | malformed requests could cause crashes | **Low** | Continue input validation hardening |
| **SSRF via Connector** | S3/Azure/Redis connectors load from URLs | Could fetch from internal endpoints | **Low** (server-side) | URL allowlisting for connectors |
| **GPU memory exposure** | KV cache contains prompt history in GPU memory | Neighboring GPU processes may read | **Low** (requires local access) | Use CUDA MPS for isolation |
| **Model weight tampering** | `update_weights_from_tensor()` hot-reloads weights | Could be used for model poisoning | **Medium** (requires API access) | Require authentication for weight update endpoints |

### Risk Severity Scale
| Severity | Definition |
|---|---|
| **High** | Exploitable without authentication, leads to data exposure or service compromise |
| **Medium** | Requires authentication or specific conditions |
| **Low** | Requires local access or extreme conditions |

## 7. AI / Agent Specific Risks

| Risk | Attack Path | Evidence | Mitigation |
|---|---|---|---|
| **Prompt Injection** | User input contains instructions to override system prompt | No input sanitization for prompt injection found | Add prompt injection detection/filtering |
| **Jailbreak via Structured Output** | Malicious grammars could leak model weights | Grammar compilation sandboxed (ThreadPoolExecutor) | Monitor for excessive grammar submissions |
| **Data Exfiltration via Logprobs** | `return_logprob=True` could leak training data | Standard LLM API feature, not a vulnerability | Standard monitoring |
| **Denial of Wallet** | Unlimited generation requests | No cost tracking, no token quotas | Add per-user token budgets |
| **Model Inversion** | Repeated queries to extract training data | Standard LLM concern, not SGLang specific | Model-level mitigation |
| **Tool Abuse (if using function calling)** | Malicious function call definitions | function_call/ module exists | Validate function schemas, add allowlists |
| **LoRA Tampering** | Untrusted LoRA adapter upload via `load_lora_adapter()` | LoRA loading from user-specified paths | Validate LoRA source, restrict to trusted paths |

## 8. Dependency Health

| Signal | Observation | Risk |
|---|---|---|
| **Update Frequency** | High — auto-bump bots for flashinfer, kernel, sglang | Low (actively maintained) |
| **Version Pinning** | Strict pins (==) for key deps (torch, flashinfer, xgrammar) | Medium (upgrade friction) |
| **Dependency Count** | ~80 direct Python deps; ~10 optional extras | Medium (attack surface) |
| **Native Dependencies** | Rust (gRPC), CUDA (flashinfer, kernels), C++ (sgl-kernel) | Medium (build complexity) |
| **GPU-specific Deps** | ~15 packages are CUDA/NVIDIA-specific | Medium (platform lock-in) |
| **Security Advisories** | Not checked (no vulnerability scanner run) | Unknown |
| **Supply Chain Attacks** | External packages: flashinfer, cutlass-dsl, tilelang from PyPI | Medium (trust in PyPI ecosystem) |

## 9. Production Security Checklist

| # | Item | Status | Action |
|---|---|---|---|
| 1 | Set API key authentication | ❌ Not by default | `--api-key <secure-random-key>` |
| 2 | Enable TLS termination | ❌ Not built-in | Configure reverse proxy (nginx/Caddy) with TLS |
| 3 | Restrict network binding | ⚠️ Default: `127.0.0.1` | Keep localhost binding; use reverse proxy for external access |
| 4 | Set production log level | ⚠️ Default: `info` | `--log-level warning` to avoid data leakage |
| 5 | Disable trust_remote_code | ❌ Not disabled by default | `--disable-trust-remote-code` for production |
| 6 | Run vulnerability scan | ❌ Not run | `pip-audit` or `safety check` on dependencies |
| 7 | Set resource limits | ⚠️ Some defaults exist | `--max-total-tokens`, `--max-running-requests` |
| 8 | Enable audit logging | ❌ Not built-in | Add request logging middleware with timestamps, IPs |
| 9 | Isolate GPU processes | ✅ CUDA MPS default | Verify process isolation in multi-tenant setups |
| 10 | Configure health checks | ✅ `/health` and `/health_generate` endpoints | Point load balancer at health endpoints |
| 11 | ZMQ IPC hardening | ⚠️ No transport security | Bind ZMQ to localhost; use Unix domain sockets if possible |
| 12 | Model access control | ❌ Not built-in | Restrict model file permissions |
| 13 | Container security | ⚠️ Dockerfiles exist | Run as non-root, use distroless base, scan images |
| 14 | Secret rotation | ❌ Manual | Implement API key rotation mechanism |

## 10. Security Verdict

### Overall Assessment: **C+ (Needs hardening for production)**

SGLang is designed as a research/production inference engine, not a security-hardened product. It assumes deployment behind a secure network boundary (firewall, API gateway). Key gaps:

1. **Authentication is opt-in** — `--api-key` flag exists but is not enforced by default
2. **No built-in authorization** — all endpoints accessible with same key
3. **No rate limiting** — vulnerable to resource exhaustion
4. **trust_remote_code risk** — off by default is safer for production
5. **No encryption at rest or in transit** — defer to infrastructure

### Commercial Deployment Readiness: **Conditional**
✅ Safe to deploy **behind a secure API gateway** with TLS termination, authentication, and rate limiting.
❌ **Do not expose directly to the internet** without security middleware.
✅ Apache 2.0 license is commercially friendly with no copyleft obligations.
⚠️ Audit NVIDIA-specific dependencies (cutlass-dsl license) for commercial redistribution.
