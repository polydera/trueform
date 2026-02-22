// ============================================================================
// Test harness — registry + helpers (no DOM, no circular imports)
// ============================================================================

export const tests = [];
let _tf = null;

export function setTf(tf) { _tf = tf; }
export function getTf() { return _tf; }

export function describe(group, fn) {
  const marker = tests.length;
  tests.push({ group, _marker: true });
  fn();
  if (tests.length === marker + 1) tests.pop();
}

export function test(name, fn) {
  tests.push({ name, fn });
}

export function log(text, cls = "") {
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = text;
  const el = document.getElementById("output");
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

export function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

export function approx(a, b, msg, eps = 1e-5) {
  assert(Math.abs(a - b) < eps, `${msg}: expected ${b}, got ${a}`);
}
