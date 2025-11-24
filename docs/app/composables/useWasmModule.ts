import type { MainModule } from "@/examples/native";

type MeshDescriptor = { url: string; filename: string };

const wasmState: { instance: MainModule | null; promise: Promise<MainModule> | null } = {
  instance: null,
  promise: null,
};

const meshCache = new Map<string, Promise<Uint8Array>>();

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
    }
  };

  return { loadWasmModule, preloadMeshes };
}
