# SGLang Code Study

Deep analysis report for [SGLang](https://github.com/sgl-project/sglang) - a high-performance LLM inference serving framework.

## Reports

| Stage | File | Topic |
|---|---|---|
| 00 | [stage_00_execution_plan.md](stage_00_execution_plan.md) | Research Execution Plan |
| 01 | [stage_01_project_intake.md](stage_01_project_intake.md) | Project Intake & Repository Map |
| 02 | [stage_02_architecture.md](stage_02_architecture.md) | Architecture & Module Boundaries |
| 03 | [stage_03_runtime.md](stage_03_runtime.md) | Runtime & Execution Lifecycle |
| 04 | [stage_04_core_features.md](stage_04_core_features.md) | Core Features & Call Chains |
| 05 | [stage_05_data_model.md](stage_05_data_model.md) | Data Model, State & Flow |
| 06 | [stage_06_api_contracts.md](stage_06_api_contracts.md) | API, SDK, CLI & Contracts |
| 07 | [stage_07_extension_points.md](stage_07_extension_points.md) | Extension Points & Secondary Dev |
| 08 | [stage_08_testing.md](stage_08_testing.md) | Testing, Quality & Debugging |
| 09 | [stage_09_security.md](stage_09_security.md) | Security, License & Supply Chain |
| 10 | [stage_10_deployment_operations_performance.md](stage_10_deployment_operations_performance.md) | Deployment, Operations & Performance |
| 11 | [stage_11_learning_path.md](stage_11_learning_path.md) | Learning Path & Code Reading Plan |
| 12 | [stage_12_ecosystem.md](stage_12_ecosystem.md) | Ecosystem, Competitive Landscape & Trends |
| 13 | [stage_13_final_report.md](stage_13_final_report.md) | **Final Synthesis Report (Chinese)** |

## Overview

- **Repository analyzed:** sgl-project/sglang
- **Research date:** 2026-05-15
- **Research depth:** DEEP - source-level static analysis
- **Method:** Static analysis only (no GPU environment available)
- **Report language:** English (stages 00-12), Chinese (stage 13 final report)

## Key Findings

1. **RadixAttention** - automatic trie-based prefix caching is the core differentiator
2. **Three-process subprocess architecture** - TokenizerManager → Scheduler → Detokenizer via ZMQ IPC
3. **12+ Mixin composition** in Scheduler for modular capability assembly
4. **28 attention backends + 191 auto-discovered models**
5. **Apache 2.0 license** - commercially friendly, maintained by LMSYS (non-profit)

## License

This report is provided for educational and reference purposes.
The analyzed project (SGLang) is licensed under Apache License 2.0.