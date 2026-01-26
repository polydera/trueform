import type { MainModule } from "@/examples/native";

type MeshDescriptor = { url: string; filename: string };

const wasmState: { instance: MainModule | null; promise: Promise<MainModule> | null } = {
  instance: null,
  promise: null,
};

const meshCache = new Map<string, Promise<Uint8Array>>();
const meshReady = new Set<string>();

type ExampleLoaderOptions<T> = {
  meshes: MeshDescriptor[];
  loading: {
    resetLoading: () => void;
    setLoadingMessage: (message: string) => void;
    failLoading: (error: unknown) => void;
    finishLoading: () => void;
  };
  skipOverlayIfCached?: boolean;
  isTornDown: () => boolean;
  createScene: (wasm: MainModule, meshFilenames: string[]) => T | null;
};

export function useWasmModule() {
  const loadWasmModule = async (): Promise<MainModule> => {
    if (wasmState.instance) {
      return wasmState.instance;
    }

    if (!wasmState.promise) {
      wasmState.promise = import("@/examples/native").then((module) => module.default());
    }

    try {
      wasmState.instance = await wasmState.promise;
      return wasmState.instance;
    } catch (error) {
      wasmState.promise = null;
      throw error;
    }
  };

  const preloadMeshes = async (wasm: MainModule, meshes: MeshDescriptor[]) => {
    for (const mesh of meshes) {
      const cached = meshCache.get(mesh.url);
      const meshPromise =
        cached ??
        fetch(mesh.url).then(async (response) => {
          if (!response.ok) {
            throw new Error(`Failed to fetch mesh at ${mesh.url}`);
          }
          const buffer = await response.arrayBuffer();
          return new Uint8Array(buffer);
        });

      meshCache.set(mesh.url, meshPromise);
      const bytes = await meshPromise;
      wasm.FS.writeFile(mesh.filename, bytes);
      meshReady.add(mesh.url);
    }
  };

  const loadExampleWithAssets = async <T>({
    meshes,
    loading,
    skipOverlayIfCached = false,
    isTornDown,
    createScene,
  }: ExampleLoaderOptions<T>): Promise<T | null> => {
    const hasWarmCache = skipOverlayIfCached && wasmState.instance && meshes.every((mesh) => meshReady.has(mesh.url));

    if (!hasWarmCache) {
      loading.resetLoading();
    } else {
      loading.finishLoading();
    }

    try {
      const wasmInstance = await loadWasmModule();
      if (isTornDown()) {
        loading.finishLoading();
        return null;
      }

      if (!hasWarmCache) {
        loading.setLoadingMessage("Fetching meshes...");
      }
      await preloadMeshes(wasmInstance, meshes);

      if (isTornDown()) {
        loading.finishLoading();
        return null;
      }

      if (!hasWarmCache) {
        loading.setLoadingMessage("Initializing renderer...");
      }
      const instance = createScene(
        wasmInstance,
        meshes.map((m) => m.filename),
      );
      loading.finishLoading();
      return instance;
    } catch (error) {
      if (!isTornDown()) {
        loading.failLoading(error);
      }
      return null;
    }
  };

  return { loadWasmModule, preloadMeshes, loadExampleWithAssets };
}
