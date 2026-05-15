# Data Model, State And Flow

*Full content available in the original report file.*

## Core Data Models

- GenerateReqInput → TokenizedGenerateReqInput → Req → ScheduleBatch → ForwardBatch
- IPC Wire Format: ZMQ PUSH/PULL between processes
- KV Cache: ReqToTokenPool, MHATokenToKVPool, RadixCache

## Data Flow Verdict

IPC-centric and stream-oriented. Strongly typed via dataclasses but weakly validated at IPC boundaries.
