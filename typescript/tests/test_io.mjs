import { describe, test, log, getTf } from "./harness.mjs";

function pickFiles(accept) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.multiple = true;
    input.onchange = () => {
      const files = Array.from(input.files);
      if (!files.length) { resolve([]); return; }
      Promise.all(
        files.map((f) => f.arrayBuffer().then((buf) => ({ name: f.name, buffer: buf })))
      ).then(resolve);
    };
    input.click();
  });
}

async function benchFile(tf, file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const bytes = new Uint8Array(file.buffer);
  const sizeMB = (bytes.length / (1024 * 1024)).toFixed(2);
  log(`  File: ${file.name} (${sizeMB} MB)`, "line-dim");

  const syncFn = ext === "stl" ? "readStl" : "readObj";
  const asyncFn = ext === "stl" ? "readStl" : "readObj";

  // Warmup
  const warmup = tf[syncFn](bytes);
  const nf = warmup.numberOfFaces;
  const np = warmup.numberOfPoints;
  warmup.delete();
  log(`  Mesh: ${nf.toLocaleString()} faces, ${np.toLocaleString()} points`, "line-dim");

  const RUNS = 5;

  // Sync benchmark
  let syncMin = Infinity;
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    const m = tf[syncFn](bytes);
    const dt = performance.now() - t0;
    syncMin = Math.min(syncMin, dt);
    m.delete();
  }
  log(`  sync:  ${syncMin.toFixed(1)} ms`, "line-bench");

  // Async benchmark
  let asyncMin = Infinity;
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    const m = await tf.async[asyncFn](bytes);
    const dt = performance.now() - t0;
    asyncMin = Math.min(asyncMin, dt);
    m.delete();
  }
  log(`  async: ${asyncMin.toFixed(1)} ms`, "line-bench");

  // Topology benchmarks
  const mesh = tf[syncFn](bytes);

  const t_fm = performance.now();
  const fm = mesh.faceMembership;
  log(`  faceMembership:     ${(performance.now() - t_fm).toFixed(1)} ms`, "line-bench");
  fm.delete();

  const t_mel = performance.now();
  const mel = mesh.manifoldEdgeLink;
  log(`  manifoldEdgeLink:   ${(performance.now() - t_mel).toFixed(1)} ms`, "line-bench");
  mel.delete();

  const t_fl = performance.now();
  const fl = mesh.faceLink;
  log(`  faceLink:           ${(performance.now() - t_fl).toFixed(1)} ms`, "line-bench");
  fl.delete();

  const t_vl = performance.now();
  const vl = mesh.vertexLink;
  log(`  vertexLink:         ${(performance.now() - t_vl).toFixed(1)} ms`, "line-bench");
  vl.delete();

  mesh.delete();
  log("");
}

describe("IO Benchmarks", () => {

  test("readStl / readObj (file dialog)", async () => {
    const tf = getTf();
    const files = await pickFiles(".stl,.obj");
    if (!files.length) { log("  No files selected, skipping.", "line-dim"); return; }

    for (const file of files) {
      await benchFile(tf, file);
    }
  });

});
