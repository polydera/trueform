# Docker-based WebAssembly builder

This folder contains a Docker image + `docker compose` service that builds the `wasm-examples` WebAssembly artifacts without installing Emscripten (or its build dependencies) on your host machine.

It is meant as a “just build the `.wasm` + JS glue” helper. It does not run the Vue app or a dev server.

## What it does

`Dockerfile` (image build time):

- Starts from `emscripten/emsdk:4.0.15` (pinned Emscripten toolchain).
- Installs build tools: `cmake`, `ninja`, `git`.
- Clones and builds `oneTBB` using the Emscripten toolchain, then installs it.
- Installs TypeScript next to the Emscripten compiler so Emscripten can generate `.d.ts` files via `--emit-tsd`.
- Builds a tiny “LTO enabled” example to warm up Emscripten’s cache for faster subsequent builds.

`docker-compose.yml`:

- Mounts the repo root into the container at `/workspace`.
- Runs `./wasm-examples/extras/build_web.sh` inside the container to configure + build `wasm-examples` with `emcmake`.
- `build_web.sh` deletes the configured build directory each run to ensure a clean rebuild.
- Copies build outputs into `wasm-examples/build/dist` on your host (via `ARTIFACTS_DIR=/workspace/wasm-examples/build/dist`).

## Prerequisites

- Docker Engine / Docker Desktop installed and running.
- Docker Compose v2 (the `docker compose` command, not `docker-compose`).
- Enough disk space for the Docker image and build output (the first build can take a while).
- Internet access during the image build (the image clones oneTBB from GitHub and installs TypeScript via npm).

Platform notes:

- The compose service sets `platform: linux/amd64`. On Apple Silicon / ARM hosts this will use emulation, which works but is slower. If you want native ARM builds, remove that line and ensure the base image/toolchain works for your setup.

## Quick start

Run from this directory (`wasm-examples/extras`):

```bash
docker compose up --build
```

Or from the repo root:

```bash
docker compose -f wasm-examples/extras/docker-compose.yml up --build
```

When it completes successfully, the artifacts are on your host in:

- `wasm-examples/build/dist`

Typical outputs include `native.wasm`, `native.js`, `native.d.ts`, and `native.js.symbols`.

## Common workflows

One-shot build (recommended over `up` if you don’t need log-following):

```bash
docker compose run --rm web-example
```

Rebuild the image from scratch (useful if you changed the Dockerfile or want to refresh caches):

```bash
docker compose build --no-cache
```

Clean build outputs on the host:

```bash
rm -rf ../build/web ../build/dist
```

## Configuration

`build_web.sh` supports a few environment variables you can override when running the container:

- `BUILD_DIR` (default: `wasm-examples/build/web`)
- `DIST_DIR` (default: `wasm-examples/build/web/dist`)
- `CMAKE_GENERATOR` (default: `Ninja`)
- `CMAKE_FLAGS` (default: disables Python/examples, sets `DIST_DIR`, uses Release builds)

Example: add extra CMake flags for the build:

```bash
docker compose run --rm \
  -e CMAKE_FLAGS="-DTF_BUILD_EXAMPLES=OFF -DTF_BUILD_PYTHON=OFF -DDIST_DIR=/workspace/wasm-examples/build/web/dist -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON" \
  web-example
```

## Troubleshooting

- **Docker cannot see your files (Windows/macOS):** ensure your repo directory/drive is shared with Docker Desktop.
- **`docker compose` not found:** update Docker Desktop / Docker Engine to a version that includes Compose v2.
- **Build fails during image build:** the image needs internet access to `git clone` oneTBB and to `npm install` TypeScript.
