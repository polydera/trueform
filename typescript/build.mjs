import { build } from "esbuild";
import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const buildDir = resolve(root, "build-wasm");

// -- WASM build (emcmake + cmake) --
if (!existsSync(resolve(buildDir, "CMakeCache.txt"))) {
  console.log("Configuring WASM build...");
  execSync(`emcmake cmake -S ${root} -B ${buildDir} -DTF_BUILD_TYPESCRIPT=ON`, {
    stdio: "inherit",
  });
}

console.log("Building WASM...");
execSync(`cmake --build ${buildDir} --target trueform_wasm --parallel`, {
  stdio: "inherit",
});

// -- TypeScript build (esbuild) --
console.log("Bundling TypeScript...");
await build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  format: "esm",
  outfile: "dist/index.js",
  sourcemap: true,
  external: ["./trueform_wasm.js"],
});
