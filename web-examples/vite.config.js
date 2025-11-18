import { fileURLToPath, URL } from 'node:url'

import {defineConfig, normalizePath} from 'vite'
import vue from '@vitejs/plugin-vue'
import glob from "fast-glob";
import fs from "fs";
import path from "path";
import vueDevTools from 'vite-plugin-vue-devtools'

export function viteBeforeServerStart(fn) {
    return {
        name: "vite-before-server-start",
        async configureServer(server) {
            await fn();
        },
    };
}

/** Plugin to run a function after the build starts. Does not run when starting dev server */
export function viteOnBuildStart(fn) {
    return {
        name: "vite-on-build-start",
        apply: "build",
        async buildStart() {
            console.log("vite-on-build-start");
            await fn();
        },
    };
}

async function copyFiles(args) {
    const { src, dest, verbose, makeDirs } = args;
    const files = await glob(src);
    await Promise.all(
        files.map(async (file) => {
            const fileName = path.basename(file);
            const destFile = path.join(dest, fileName);
            if (makeDirs) await fs.promises.mkdir(path.dirname(destFile), { recursive: true });
            if (fs.existsSync(destFile)) {
                const srcStats = await fs.promises.stat(file);
                const destStats = await fs.promises.stat(destFile);
                if (srcStats.mtimeMs <= destStats.mtimeMs) {
                    if (verbose) console.log(`Skipping ${file} - not modified`);
                    return;
                }
            }
            await fs.promises.copyFile(file, destFile);
            if (verbose) console.log(`Copied ${file} to ${destFile}`);
        }),
    );
}

const copyDeps = async () => {
    const copy1 = copyFiles({
        src: normalizePath(path.resolve(__dirname, `./src/webAssembly/build/dist/*.wasm`)),
        dest: normalizePath(path.resolve(__dirname, `./public`)),
        makeDirs: true,
    });
    await Promise.all([copy1]);
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
      // basicSSL(),
    vueDevTools(),
    viteBeforeServerStart(copyDeps),
    viteOnBuildStart(copyDeps),
  ],
    server: {
        // https: true,
        headers: {
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "cross-origin",
        },
    },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
})
