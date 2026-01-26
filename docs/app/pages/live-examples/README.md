### Shared loader helper

`useWasmModule.loadExampleWithAssets` wraps the common boot steps for live examples:

1) `resetLoading` / WASM load
2) mesh prefetch
3) scene init via `createScene`
4) `finishLoading` or `failLoading`

Each page passes its own containers and example class into `createScene`, plus `isTornDown` to stop mid-flight. Reuse this pattern to keep loading UX consistent.

If `skipOverlayIfCached` is true and the WASM + meshes are already ready, the overlay is skipped to avoid flicker on subsequent example pages.
