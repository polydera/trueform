import { defineNuxtModule } from "nuxt/kit";
import { glob } from "fast-glob"
import { mkdir, copyFile, stat } from "fs/promises";
import { dirname, join, basename } from "pathe";

export default defineNuxtModule({
  meta: {
    name: "copy-files",
  },

  setup(options, nuxt) {
    async function copyAll() {
      // Copy WASM module
      const wasmFile = {
        from: "../wasm-examples/build/dist/native.wasm",
        to: "public/native.wasm",
      };

      try {
        await stat(wasmFile.from);
      } catch {
        throw new Error(`❌ Required file is missing in ${wasmFile.from}: ${wasmFile.to}`);
      }

      await mkdir(dirname(wasmFile.to), { recursive: true });
      await copyFile(wasmFile.from, wasmFile.to);

      // Copy all files from build/dist to app/examples
      const distDir = "../wasm-examples/build/dist";
      const allFiles = await glob(`${distDir}/**/*`, {
        absolute: false,
        onlyFiles: true
      });

      for (const file of allFiles) {
        const fileName = basename(file);
        const destPath = join("app/examples/", fileName);

        await mkdir(dirname(destPath), { recursive: true });
        await copyFile(file, destPath);
      }

      // Copy STL meshes from benchmarks
      const stlFiles = await glob("../benchmarks/data/dragon-*.stl", {
        absolute: false,
        onlyFiles: true
      });

      await mkdir("public/stl", { recursive: true });

      for (const file of stlFiles) {
        const fileName = basename(file);
        await copyFile(file, join("public/stl", fileName));
      }
    }

    // Before dev server starts
    nuxt.hooks.hook("prepare:types", async () => {
      await copyAll();
    });
  },
});
