# API, SDK, CLI And Contracts

*Full content available in the original report file.*

## Public Surface Summary

- HTTP API routes: ~40+ (Native + OpenAI + Ollama + Anthropic + SageMaker + Vertex AI)
- CLI commands: 3 (serve, generate, version) + killall
- Python SDK: sglang.lang API + Engine API
- gRPC API: Protobuf-defined (Beta)

## Contract Risks

- No API versioning
- No formal deprecation policy
- No API key enforcement by default
