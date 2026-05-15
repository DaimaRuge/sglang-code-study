# Runtime And Execution Lifecycle

*Full content available in the original report file.*

## Key Findings

- Three-process pipeline: Main Process + Scheduler Subprocess (GPU) + Detokenizer Subprocess
- Startup chain: CLI → Engine → fork Scheduler + Detokenizer → ZMQ IPC → HTTP Server
- Plugin registration via setuptools entry_points and HookRegistry
- Graceful shutdown via SIGTERM/SIGINT signal handling
- SubprocessWatchdog for liveness monitoring

## Runtime Verification

All runtime conclusions based on static analysis only (no GPU available for dynamic testing).
