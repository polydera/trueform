import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const root = join(__dirname, "..");

const MIME = {
  ".html": "text/html",
  ".js":   "application/javascript",
  ".mjs":  "application/javascript",
  ".wasm": "application/wasm",
  ".css":  "text/css",
};

const server = createServer(async (req, res) => {
  // COOP + COEP headers required for SharedArrayBuffer
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");

  if (req.url === "/") {
    res.writeHead(302, { Location: "/tests/test.html" });
    res.end();
    return;
  }
  const filePath = join(root, req.url);
  const ext = extname(filePath);

  try {
    const data = await readFile(filePath);
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end("Not found: " + req.url);
  }
});

server.listen(3000, () => {
  console.log("Serving at http://localhost:3000  (COOP+COEP enabled)");
});
