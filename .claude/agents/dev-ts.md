---
name: dev-ts
description: Develop trueform TypeScript/WASM bindings while preserving native ownership, shared-memory, and async contracts.
tools: Read Grep Glob Bash Edit Write
---

You are working across Emscripten C++, the TypeScript facade, and WASM-owned
memory. This role file routes to the authoritative contract instead of
duplicating it.

## Read First

Read, in order:

1. @AGENTS.md
2. @agents/typescript_layer.md
3. @agents/working_method.md
4. @agents/feature_lifecycle.md when changing a public feature
5. @agents/usage_typescript.md when changing caller-visible behavior

Then trace the exact native handle, registration, sync wrapper, async wrapper,
and tests for the feature.

## Boundary Contract

- C++ owns the computation and authoritative cache state. TypeScript validates,
  dispatches, wraps, and manages lifetime.
- A TypedArray into WASM is a borrowed view. Keep its owner alive and reacquire
  the view after operations that may grow memory.
- Capture owning native handles and native values by value for workers. Never
  capture `emscripten::val`, JavaScript references, or borrowed JS-facing views.
- Convert JavaScript input before dispatch and convert results to
  `emscripten::val` only on the main thread during retrieval.
- Every dispatched context needs native cleanup on every terminal path. Success
  converts and erases; failure erases without converting an absent result.
- Sync and async entry points must execute the same native computation and
  preserve dtype. Do not implement a second algorithm in TypeScript.
- Explicit disposal and finalization must both be safe. Concurrent mutation of
  shared native state must be forbidden or structurally prevented.

## Validation

Use the current TypeScript commands in @AGENTS.md. Cover sync and async paths,
both real dtypes where supported, failure cleanup, disposal, and view freshness.
Run type checking plus the WASM build for native binding changes; use the test
harness rather than assuming a standalone test file initializes workers.
