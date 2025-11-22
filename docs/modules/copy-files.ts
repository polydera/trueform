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
      const wasmFile = {
        from: "../web-examples/src/webAssembly/build/dist/native.wasm",
        to: "public/native.wasm",
      };

      try {
        await stat(wasmFile.from);
      } catch {
        throw new Error(`❌ Required file is missing in ${wasmFile.from}: ${wasmFile.to}`);
      }

      await mkdir(dirname(wasmFile.to), { recursive: true });
      await copyFile(wasmFile.from, wasmFile.to);

      // Copy all files from build/dist except .wasm files
      const distDir = "../web-examples/src/webAssembly/build/dist";
      const allFiles = await glob(`${distDir}/**/*`, {
        absolute: false,
        onlyFiles: true
      });

      const filesToCopy = allFiles.filter(file => !file.endsWith('.wasm'));

      for (const file of filesToCopy) {
        const fileName = basename(file);
        const destPath = join("app/examples/", fileName);

        await mkdir(dirname(destPath), { recursive: true });
        await copyFile(file, destPath);
      }
    }

    // Before dev server starts
    nuxt.hooks.hook("prepare:types", async () => {
      await copyAll();
    });
  },
});
