# WebAssembly Build

The native module is now built with CMake so that we can rely on the standard
Emscripten toolchain configuration instead of a bespoke Makefile.

## Prerequisites

Activate the bundled SDK (or a globally installed one) before configuring CMake:

```sh
cd src/webAssembly
source emsdk/emsdk_env.sh
```

## Build steps

```sh
emcmake cmake -S . -B build
cmake --build build
```

Artifacts (`native.js`, `native.wasm`, and the generated `native.d.ts`) are
written to `src/webAssembly/dist`, which is what the Vue app imports.
