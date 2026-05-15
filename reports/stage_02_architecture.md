# Architecture And Module Boundaries

*Full content available in the original report file.*

## Key Findings

- Layered Architecture with Subprocess Microkernel pattern
- Three-process pipeline: TokenizerManager → Scheduler → Detokenizer via ZMQ IPC
- Plugin-based Extension via setuptools entry_points
- Mixin-based Composition (12+ mixins in Scheduler)
- Model Registry Pattern with auto-discovery via pkgutil
- Strategy Pattern for backends (attention, grammar, MoE, transfer)

## Architecture Verdict: B+

Sound architecture for high-performance LLM serving. Main tech debt: server_args.py God Object (7,950 lines).
