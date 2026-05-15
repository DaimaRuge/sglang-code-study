# Project Intake And Repository Map

## 1. Project Identity Card

| Field | Value | Evidence | Confidence |
|---|---|---|---|
| Project Name | SGLang | README.md:1, pyproject.toml:6 | High |
| Owner | sgl-project (LMSYS) | README.md:81, GitHub | High |
| Repository URL | https://github.com/sgl-project/sglang | README.md:169 | High |
| Local Path | d:\codeStudy\sglang | Filesystem | High |
| License | Apache 2.0 | LICENSE:1-3 | High |
| Version | Dynamic (setuptools_scm based on git tag) | pyproject.toml:7, version.py:13-20 | High |
| Latest Release | v0.5 (inferred from changelogs/PRs) | README news mentions v0.2-0.4 | Medium |
| Python Requirement | >=3.10 | pyproject.toml:10 | High |
| Total Commits | 12,649 | `git rev-list --count HEAD` | High |
| Latest Commit | 2026-05-15: Enable SGLANG_OPT_FP8_WO_A_GEMM by default | `git log -1` | High |
| Primary Language | Python (with Rust for gRPC, C++ for CUDA kernels) | pyproject.toml, 3rdparty/, rust/, sgl-kernel/ | High |
| Total Python Files | 1,976 in main package | `find . -name '*.py' \| wc -l` | High |

*Note: Full content available in the original report file.*
