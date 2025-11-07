# Docker Compose helpers

The files in this directory let you build and serve the WebAssembly examples from a containerized toolchain. The `misc/docker-compose.yml` file exposes the `web-example` service, which runs `misc/build_web.sh` inside the container and (optionally) serves the resulting demo via `emrun`.

## Required environment variables

`docker compose` reads the following variables from your shell or a `.env` file that lives next to the compose file:

| Variable            | Purpose                     | Default        |
|---------------------|-----------------------------|----------------|
| `TF_SERVE_PORT`     | Host port used by `emrun`   | `8080`         |
| `TF_EXAMPLE_TARGET` | CMake target to build/serve | `web_example1` |

Example `.env` file:

```env
# misc/.env
TF_SERVE_PORT=9000
TF_EXAMPLE_TARGET=web_example1
```

You can also set them inline when invoking compose:

```bash
TF_SERVE_PORT=9000 TF_EXAMPLE_TARGET=web_example1 docker compose -f misc/docker-compose.yml up
```

## Bringing the service up

```bash
# From the repo root
docker compose -f misc/docker-compose.yml up --build
```

The first run builds the image (including Emscripten) and compiles the requested example. When `SERVE=1` (the default inside `build_web.sh`), the example is hosted at `http://localhost:${TF_SERVE_PORT}`. Use `docker compose down` to stop the service when you are done.
