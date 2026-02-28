import { build } from "esbuild";
import { execSync } from "node:child_process";
import { copyFileSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const buildDir = resolve(root, process.env.TF_WASM_BUILD_DIR || "build-wasm");

// -- WASM build (emcmake + cmake) --
if (!existsSync(resolve(buildDir, "CMakeCache.txt"))) {
  console.log("Configuring WASM build...");
  execSync(`emcmake cmake -S ${root} -B ${buildDir} -DTF_BUILD_TYPESCRIPT=ON -DCMAKE_BUILD_TYPE=Release`, {
    stdio: "inherit",
  });
}

console.log("Building WASM...");
execSync(`cmake --build ${buildDir} --target trueform_wasm --parallel`, {
  stdio: "inherit",
});

// -- Stamp version from CMake --
const cache = readFileSync(resolve(buildDir, "CMakeCache.txt"), "utf-8");
const version = cache.match(/^CMAKE_PROJECT_VERSION:STATIC=(.+)$/m)?.[1];
if (version) {
  const pkgPath = resolve(__dirname, "package.json");
  const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
  if (pkg.version !== version) {
    pkg.version = version;
    writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
    console.log(`Stamped version ${version}`);
  }
}

// -- Copy license files --
for (const f of ["LICENSE", "LICENSE.noncommercial"]) {
  copyFileSync(resolve(root, f), resolve(__dirname, f));
}

// -- TypeScript compile (tsc) --
console.log("Compiling TypeScript...");
execSync("npx tsc", { stdio: "inherit", cwd: __dirname });

// -- Bundle (esbuild) --
console.log("Bundling...");
await build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  format: "esm",
  outfile: "dist/index.js",
  sourcemap: true,
  external: ["./trueform_wasm.js"],
});
