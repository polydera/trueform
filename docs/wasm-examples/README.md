# WebAssembly Build

The native module is now built with CMake so that we can rely on the standard
Emscripten toolchain configuration instead of a bespoke Makefile.

## Prerequisites

Activate the bundled SDK (or a globally installed one) before configuring CMake:

```sh
cd wasm-examples
source emsdk/emsdk_env.sh
```

### TypeScript support

The build runs Emscripten with `--emit-tsd`, which shells out to the TypeScript
compiler that lives next to `em++`. Configuring the bundled SDK installs
TypeScript automatically. If you are using a global SDK (or a read-only one),
ensure that `tsc` exists under `<emscripten root>/node_modules/.bin` by running:

```sh
cd <path-to-emscripten>
npm install --no-save --package-lock false typescript@5.9.3
```

## Build steps

```sh
emcmake cmake -S . -B build
cmake --build build
```

Artifacts (`native.js`, `native.wasm`, and the generated `native.d.ts`) are
written to `wasm-examples/build/dist`, which is what the Vue app imports.
