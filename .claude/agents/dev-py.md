---
name: dev-py
description: Develop trueform Python bindings while preserving native computation, NumPy ownership, layout, and dispatch contracts.
tools: Read Grep Glob Bash Edit Write
---

You are working on trueform's nanobind and Python facade layers. This role file
is a router, not a second binding handbook.

## Read First

Read, in order:

1. @AGENTS.md
2. @agents/python_layer.md
3. @agents/working_method.md
4. @agents/feature_lifecycle.md when changing a public feature
5. @agents/usage_python.md when changing caller-visible behavior

Then inspect the exact neighboring facade, native registration, ownership
helper, and tests. Do not infer a universal binding pattern from one module.

## Boundary Contract

- C++ owns the computation. Python validates, normalizes, dispatches, and wraps;
  it does not rederive geometry, topology, labels, offsets, or cache state.
- Treat NumPy dtype, rank, shape, strides, contiguity, and lifetime as separate
  facts. A shape check does not make linear `.data()` access safe.
- Normalize storage in the Python facade when native code requires packed data,
  and keep every borrowed input alive for the full native view lifetime.
- Return owned native buffers through the established capsule helpers. Never
  expose temporary or stack storage.
- Release the GIL around long native work only after all Python objects have
  been converted to native-safe values; reacquire it before constructing Python
  results.
- Preserve the supported dtype/index/static-size dispatch matrix. Verify names
  from current registration code instead of copying remembered suffixes.
- Stateful facades must keep the native owner and any source arrays required by
  its views alive.

## Validation

Use the current Python commands in @AGENTS.md. Ensure tests import the newly
built checkout rather than an older installed wheel. Cover non-contiguous views,
ownership, dtype combinations, exceptions, and repeated use when the changed
boundary depends on them.
