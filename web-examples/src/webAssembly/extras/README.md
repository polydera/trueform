# Docker Compose helpers

Builds the WebAssembly examples using Docker Compose. This is useful if you do not want to install Emscripten and other dependencies on your host machine.

## Bringing the service up

```bash
# From the Dockerfile directory
docker compose up --build
```

The first run builds the image and compiles the library. Artifacts are copied to `web-examples/src/webAssembly/build/dist` folder.
