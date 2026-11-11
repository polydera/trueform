#!/usr/bin/env python3
"""The law scan: the textual half of the house's concurrency and test laws,
run beside the portability scan as the first item of every gate. Exit code
1 on any violation; nothing is baselined — a hit is a defect.

  L1  a wait on another thread's progress in the library: std::this_thread
      ::yield, a load loop on an atomic (`for (;;)` / `while` whose body
      loads an atomic and breaks on its value), a spin, a mutex, a condition
      variable — concurrency comes from structural separation or phase
      boundaries, never from one thread waiting on another
  L2  a test that oversubscribes: tbb::task_arena or tbb::global_control
      given a literal worker count other than 1 — a ladder is derived from
      the machine (max_concurrency / hardware_concurrency) and capped by it
  L3  a test that enters an internal parallel scheme by hand (the Delaunay
      build's claim bands) instead of the public builder and its policies
  L4  a reference library named in a header's prose — the contract stands
      on its own
  L5  a standard hash container in the library: std::unordered_set / map /
      multiset / multimap — a bounded identity is dense, and where a hash
      container is the only choice it is tf::hash_set / tf::hash_map
  L6  a std::vector the library holds itself — the house allocator comes
      with tf::core::std_vector; only what a caller hands in or receives
      back is a plain std::vector, and those files are named in VECTOR_OK
"""
import os
import glob
import re
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

bad = []

library = [f for f in glob.glob("include/trueform/**/*.hpp", recursive=True)
           if "/external/" not in f]
tests = glob.glob("tests/**/*.cpp", recursive=True)


def code_text(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text,
                  flags=re.S)
    return "\n".join(ln.split("//")[0] for ln in text.split("\n"))


def code_lines(path):
    for i, ln in enumerate(code_text(path).split("\n"), 1):
        yield i, ln


WAIT = re.compile(r"std::this_thread::yield|std::mutex|std::shared_mutex|"
                  r"std::condition_variable|spin_mutex|spin_rw_mutex|"
                  r"queuing_mutex|std::atomic_flag|_mm_pause|__builtin_ia32_pause")
LOOP_OPEN = re.compile(r"\b(for\s*\(\s*;\s*;\s*\)|while\s*\(\s*true\s*\)|while\s*\(\s*1\s*\))")
for f in library:
    lines = list(code_lines(f))
    for i, ln in lines:
        m = WAIT.search(ln)
        if m:
            bad.append(f"{f}:{i} L1: '{m.group(0)}' -- a wait on another thread's progress")
    # a bare loop whose body loads an atomic and breaks on it
    for k, (i, ln) in enumerate(lines):
        if not LOOP_OPEN.search(ln):
            continue
        body = " ".join(l for _, l in lines[k:k + 12])
        if re.search(r"\.load\s*\(", body) and re.search(r"\bbreak\b", body) \
                and not re.search(r"compare_exchange", body):
            bad.append(f"{f}:{i} L1: a loop reading an atomic until another thread changes it")

# A count is read off the statement that states it, from the opening
# parenthesis to the balanced closing one, so an argument of its own —
# `static_cast<int>(w)` — and a statement wrapped over lines are both seen.
ARENA_OPEN = re.compile(r"\btask_arena\s*(?:\w+\s*)?\(")
CONTROL_OPEN = re.compile(r"\bglobal_control\s*(?:\w+\s*)?\(")
LADDER = re.compile(r"\b(worker_ladder|capped_workers|machine_workers)\s*\(")
CAST = re.compile(r"^(?:static_cast\s*<[^<>]*>|int|unsigned|long|std::size_t)"
                  r"\s*\(\s*(.*?)\s*\)$", re.S)


def statement_arguments(text, opening):
    """The arguments of the call whose '(' is at `opening`, split at the
    call's own commas, or None where the parentheses do not close."""
    depth, start, out = 0, opening + 1, []
    for k in range(opening, len(text)):
        if text[k] in "([{":
            depth += 1
        elif text[k] in ")]}":
            depth -= 1
            if depth == 0:
                out.append(text[start:k])
                return out
        elif text[k] == "," and depth == 1:
            out.append(text[start:k])
            start = k + 1
    return None


for f in tests:
    text = code_text(f)
    bound = set()
    # a count is lawful when it is the literal 1 or an identifier bound
    # from the machine through the test ladder helpers in this file
    for m in re.finditer(r"\b(?:auto|int|std::size_t|unsigned|const auto|const int|const std::size_t)\s+(\w+)\s*[=:]\s*[^;{]*?\b(?:worker_ladder|capped_workers|machine_workers)\s*\(", text):
        bound.add(m.group(1))
    for m in re.finditer(r"\bfor\s*\(\s*(?:auto|int|std::size_t|unsigned)\s+(\w+)\s*:\s*[^)]*?\b(?:worker_ladder|capped_workers|machine_workers)\s*\(", text):
        bound.add(m.group(1))
    def lawful(expr):
        expr = " ".join(expr.split())
        while True:
            inner = CAST.sub(r"\1", expr)
            if inner == expr:
                break
            expr = inner
        if expr == "1":
            return True
        if LADDER.search(expr):
            return True
        return expr in bound
    def at(position):
        return text.count("\n", 0, position) + 1
    for m in ARENA_OPEN.finditer(text):
        args = statement_arguments(text, m.end() - 1)
        if args is None:
            bad.append(f"{f}:{at(m.start())} L2: a task_arena statement whose parentheses the scan cannot close")
        elif args[0].strip() and not lawful(args[0]):
            bad.append(f"{f}:{at(m.start())} L2: task_arena({' '.join(args[0].split())}) -- a worker count not bound from the machine through tf::test's ladder")
    for m in CONTROL_OPEN.finditer(text):
        args = statement_arguments(text, m.end() - 1)
        if args is None:
            bad.append(f"{f}:{at(m.start())} L2: a global_control statement whose parentheses the scan cannot close")
        elif len(args) == 2 and "max_allowed_parallelism" in args[0] \
                and not lawful(args[1]):
            bad.append(f"{f}:{at(m.start())} L2: global_control(max_allowed_parallelism, {' '.join(args[1].split())}) -- a worker count not bound from the machine through tf::test's ladder")

INTERNAL = re.compile(r"\b(insert_tet_claim_band|run_tet_claim_stream|grow_tet_claim_pools|"
                      r"open_tet_claim_streams|bootstrap_tet_claims|finish_tet_claims|"
                      r"prepare_tet_claim_sites|redistribute_tet_claim_rows|acquire_tet_cell|"
                      r"locate_claimed_tet|discover_claimed_tet_cavity|release_tet_cells|"
                      r"release_tet_claim_pools|measure_tet_claim_density|"
                      r"tet_claim_rows_asked)\b")
for f in tests:
    for i, ln in code_lines(f):
        m = INTERNAL.search(ln)
        if m:
            bad.append(f"{f}:{i} L3: '{m.group(0)}' entered by hand from a test -- claims are stated through the builder and its policies")

NAMES = re.compile(r"\b(TetGen|CGAL|Geogram|TetWild|fTetWild|libigl|MeshLib|HXT|Gmsh|gmsh|Shewchuk|Manifold3D|EMBER)\b")
for f in library:
    for i, ln in enumerate(open(f, encoding="utf-8"), 1):
        m = NAMES.search(ln)
        if m:
            bad.append(f"{f}:{i} L4: '{m.group(0)}' named in a header")

HASHED = re.compile(r"\bstd::unordered_(?:multi)?(?:set|map)\b|#\s*include\s*<unordered_(?:set|map)>")
for f in library:
    for i, ln in code_lines(f):
        m = HASHED.search(ln)
        if m:
            bad.append(f"{f}:{i} L5: '{m.group(0)}' -- the library's hash containers are tf::hash_set and tf::hash_map")

# The files whose std::vector is a caller's own container, and the ones that
# state the house type or bind any allocator. Each is allowed, with its
# reason.
VECTOR_OK = {
    "include/trueform/core/memory.hpp": "states the alias",
    "include/trueform/core/allocate.hpp": "any allocator binds",
    "include/trueform/core/reallocate.hpp": "any allocator binds",
    "include/trueform/core/algorithm/generate_offset_blocks.hpp":
        "any allocator binds",
    "include/trueform/core/local_vector.hpp": "the caller aggregates here",
    "include/trueform/csg/expression/expr.hpp": "the caller's children",
    "include/trueform/csg/expression/make_children.hpp":
        "builds expr's children",
    "include/trueform/csg/expression/any_of.hpp": "builds expr's children",
    "include/trueform/csg/expression/all_of.hpp": "builds expr's children",
    "include/trueform/csg/expression/compile.hpp": "reads expr's children",
    "include/trueform/csg/expression/compile_cluster_node.hpp":
        "reads expr's children",
    "include/trueform/io/obj_file.hpp": "the read's returned names",
    "include/trueform/io/obj/obj_tables.hpp": "moved into obj_file",
    "include/trueform/io/obj/resolve_obj_labels.hpp": "moves obj_file's names",
    "include/trueform/reindex/range.hpp": "the caller's vector output",
    "include/trueform/reindex/split_into_components.hpp":
        "returns the caller's vectors",
    "include/trueform/reindex/split_into_domains.hpp":
        "returns the caller's vectors",
    "include/trueform/topology/half_edges_like.hpp":
        "the caller's ring container",
}
VECTORED = re.compile(r"\bstd::vector\s*<")
for f in library:
    if f.replace(os.sep, "/") in VECTOR_OK:
        continue
    for i, ln in code_lines(f):
        if VECTORED.search(ln.replace("tf::core::std_vector", "")):
            bad.append(f"{f}:{i} L6: 'std::vector' -- the library holds its "
                       "own vectors in tf::core::std_vector")

for b in bad:
    print("VIOLATION:", b)
print("law scan:", "FAILED" if bad else "clean", f"({len(bad)} violations)")
sys.exit(1 if bad else 0)
