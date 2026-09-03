# Internal Allocator Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every heap allocation trueform *owns* through one swappable, user-facing seam (`tf::allocate`/`tf::deallocate`), behaviour-neutral now (backend = `operator new/delete`), so a fast allocator can later be dropped in without users preloading anything.

**Architecture:** A new `core/memory.hpp` exposes `tf::allocate(bytes)→void*`, `tf::deallocate(void*)`, a std-compatible `tf::allocator<T>`, and `tf::core::std_vector<T>`. `tf::buffer`, `tf::hash_map/set`, the one `std::queue`, `finder`'s scratch, and internal-only `std::vector`s route through it. The 14 sites that `release()` a buffer to VTK/numpy switch their free to `tf::deallocate`. User-returned `std::vector`s and interop overloads stay on the default allocator.

> **Names (collision-checked):** `tf::allocate`/`tf::deallocate` (top-level `namespace tf`) are **free** and are our user-facing seam. Already taken (and NOT reused): `tf::core::allocate` (the 2-arg container-resize helpers in `core/allocate.hpp`, a different namespace) and `tf::vector` (the linear-algebra vector). The internal vector alias is therefore `tf::core::std_vector<T>`.

**Tech Stack:** Header-only C++17, `tf::buffer`/`tf::small_vector`, ska2 flat hash, Catch2 (`build-tests`), VTK (`vtk/`), nanobind (`python/`), CMake.

**Spec:** `docs/superpowers/specs/2026-06-19-internal-allocator-routing-design.md`

**Worktree/branch:** `/Users/ziga/trueform-allocator` on `feature/internal-allocator`.

**Coding standards (dev-cpp, enforce on every file):** `snake_case`; `make_` factories; leading-underscore private members; trailing return types; `#pragma once` + the XLAB copyright header (copy from any sibling header); namespace close comments; no verbose comments (WHY only, when non-obvious).

**Test harness:** the repo builds Catch2 tests into `build-tests/`. The per-module pattern is `cmake --build build-tests --target trueform_<module>_tests && build-tests/tests/<module>/trueform_<module>_tests "[tag]"`. New test files are registered by adding them to that module's test `CMakeLists.txt` (follow the existing entries verbatim). Before Task 1, run `cmake --build build-tests --target trueform_core_tests` once to confirm the core test target name and path; if it differs, use the actual name throughout.

---

## File Structure

**New:**
- `include/trueform/core/memory.hpp` — `tf::allocate`, `tf::deallocate`, `tf::allocator<T>`, `tf::core::std_vector<T>`. Single allocation surface.
- `tests/core/test_memory.cpp` — unit tests for the seam.

**Modified:**
- `include/trueform/core/buffer.hpp` — storage `unique_ptr<T[]>`/`new T[]` → `tf::allocate`/`tf::deallocate`.
- `include/trueform/core/hash_map.hpp`, `hash_set.hpp` — default the allocator template arg to `tf::allocator`.
- `include/trueform/cut/arrangements/propagate_inclusion_bits.hpp` — `std::queue<Index>` → `tf::buffer<Index>` + head index.
- `include/trueform/topology/components/finder.hpp` — `new std::atomic<label_t>[]` → `tf::allocate`.
- internal-only `std::vector` sites → `tf::vector` (enumerated in Task 5).
- `vtk/src/core/make_vtk_array.cpp`, `make_vtk_points.cpp`, `make_vtk_normals.cpp`, `make_vtk_cells.cpp`, `vtk/include/trueform/vtk/core/make_vtk_cells.hpp` — `VTK_DATA_ARRAY_DELETE` → `USER_DEFINED` + `tf::deallocate`.
- `python/include/trueform/python/util/make_numpy_array.hpp`, `python/include/trueform/python/spatial/gather_ids.hpp` — capsule free → `tf::deallocate`.
- docs: `docs/content/cpp/2.modules/01.core.md`, `docs/content/cpp/5.examples/1.mesh-assembly.md`, `CLAUDE.md`, `ARTICLE_DATA_PATTERNS.md`, `STL_ARTICLE_REFERENCE.md`.

**Ordering rationale:** Task 1 adds the seam (nothing depends on behaviour). Tasks 2–5 are behaviour-neutral internal routings, each a green commit. Task 6 is the **atomic** buffer+frees change (the release-contract flip). Task 7 docs.

---

## Task 1: The allocation seam (`core/memory.hpp`)

**Files:**
- Create: `include/trueform/core/memory.hpp`
- Create + register: `tests/core/test_memory.cpp` (add to `tests/core/CMakeLists.txt` following an existing entry)

- [ ] **Step 1: Write the failing test.** `tests/core/test_memory.cpp`:

```cpp
#include <trueform/core/memory.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <numeric>

TEST_CASE("tf::allocate/tf::deallocate round-trips", "[memory]") {
  void *p = tf::allocate(64 * sizeof(int));
  REQUIRE(p != nullptr);
  auto *ints = static_cast<int *>(p);
  std::iota(ints, ints + 64, 0);
  REQUIRE(ints[63] == 63);
  tf::deallocate(p); // no leak under ASan, no crash
}

TEST_CASE("tf::allocator drives a std::vector", "[memory]") {
  std::vector<int, tf::allocator<int>> v;
  for (int i = 0; i < 1000; ++i) v.push_back(i);
  REQUIRE(v.size() == 1000);
  REQUIRE(v.back() == 999);
  tf::core::std_vector<double> w(10, 1.5);
  REQUIRE(w[9] == 1.5);
  // allocators of any T compare equal (stateless)
  REQUIRE(tf::allocator<int>{} == tf::allocator<int>{});
}
```

- [ ] **Step 2: Run it, confirm it fails to compile** (`memory.hpp` does not exist).

Run: `cmake --build build-tests --target trueform_core_tests`
Expected: FAIL — `trueform/core/memory.hpp: No such file`.

- [ ] **Step 3: Create `include/trueform/core/memory.hpp`** (copy the XLAB copyright header from a sibling, e.g. `core/buffer.hpp`):

> **Naming (collision-checked):** `tf::allocate`/`tf::deallocate` (top-level) are free — they are the seam. `tf::core::allocate` (the 2-arg container-resize helpers, a different namespace) and `tf::vector` (linalg) are taken and NOT reused; the internal vector alias is `tf::core::std_vector`.

```cpp
#pragma once

#include <cstddef>
#include <new>
#include <vector>

namespace tf {

/// User-facing allocation seam. Default backend is global operator new/delete;
/// swapping these two bodies (e.g. to tbbmalloc) re-points every trueform-owned
/// allocation. void*/bytes shaped so deallocate needs no type and `&tf::deallocate`
/// is a `void(*)(void*)` for C free-callbacks (VTK user-defined delete, nanobind
/// capsules). Distinct from tf::core::allocate (the 2-arg container-resize
/// helpers in core/allocate.hpp). Not noexcept-qualified for callback compat;
/// the bodies never throw.
inline auto allocate(std::size_t bytes) -> void* { return ::operator new(bytes); }

inline auto deallocate(void *p) -> void { ::operator delete(p); }

/// std::allocator-compatible adapter over the seam. Stateless; always equal.
template <typename T> struct allocator {
  using value_type = T;

  allocator() noexcept = default;
  template <typename U> allocator(const allocator<U> &) noexcept {}

  auto allocate(std::size_t n) -> T* {
    return static_cast<T *>(tf::allocate(n * sizeof(T)));
  }
  auto deallocate(T *p, std::size_t) noexcept -> void { tf::deallocate(p); }

  template <typename U>
  auto operator==(const allocator<U> &) const noexcept -> bool { return true; }
  template <typename U>
  auto operator!=(const allocator<U> &) const noexcept -> bool { return false; }
};

/// The one internal vector. `tf::vector` is the linear-algebra vector, so this
/// lives in tf::core and is internal-only.
namespace core {
template <typename T> using std_vector = std::vector<T, tf::allocator<T>>;
} // namespace core

} // namespace tf
```

- [ ] **Step 4: Run the tests, confirm pass.**

Run: `cmake --build build-tests --target trueform_core_tests && build-tests/tests/core/trueform_core_tests "[memory]"`
Expected: PASS (2 test cases). If ASan is configured, no leak reports.

- [ ] **Step 5: Commit.**

```bash
git add include/trueform/core/memory.hpp tests/core/test_memory.cpp tests/core/CMakeLists.txt
git commit -m "feat(core): tf::allocate/deallocate + tf::allocator/vector seam"
```

---

## Task 2: `hash_map`/`hash_set` default to `tf::allocator`

**Files:**
- Modify: `include/trueform/core/hash_map.hpp:17`, `include/trueform/core/hash_set.hpp:17`
- Test: `tests/core/test_memory.cpp` (append)

The aliases are `using hash_map = ska2::flat_hash_map<T0, T1, Ts...>;` and `using hash_set = ska2::flat_hash_set<T, Ts...>;`. ska2's `flat_hash_map<K, V, Hash, Equal, Alloc>` takes the allocator as its 5th template parameter (4th for the set: `flat_hash_set<K, Hash, Equal, Alloc>`). Inject `tf::allocator` as the *default* so existing 2-arg / 1-arg uses pick it up, while callers can still override.

- [ ] **Step 1: Write the failing test** (append to `tests/core/test_memory.cpp`):

```cpp
#include <trueform/core/hash_map.hpp>
#include <trueform/core/hash_set.hpp>

TEST_CASE("tf::hash_map/set use tf::allocator by default", "[memory]") {
  static_assert(std::is_same_v<tf::hash_map<int, int>::allocator_type,
                               tf::allocator<std::pair<int, int>>>,
                "hash_map must default to tf::allocator");
  static_assert(std::is_same_v<tf::hash_set<int>::allocator_type,
                               tf::allocator<int>>,
                "hash_set must default to tf::allocator");
  tf::hash_map<int, int> m;
  for (int i = 0; i < 500; ++i) m[i] = i * 2;
  REQUIRE(m.at(499) == 998);
  tf::hash_set<int> s{1, 2, 3, 2, 1};
  REQUIRE(s.size() == 3);
}
```

> Note: ska2's `value_type`/`allocator_type` aliases — confirm the exact `allocator_type` ska2 exposes (it may rebind to `std::pair<const int,int>` or its internal node type). Read `include/trueform/core/external/hash_map.hpp` for the actual `value_type`/`allocator_type` and match the `static_assert` to it. If ska2 stores `std::pair<K,V>` (not `const K`), use that. Do not guess — read it.

- [ ] **Step 2: Run it, confirm it fails** (default is ska2's `std::allocator`, not `tf::allocator`).

Run: `cmake --build build-tests --target trueform_core_tests`
Expected: FAIL — `static_assert` fires (allocator is `std::allocator`).

- [ ] **Step 3: Edit the aliases.** `include/trueform/core/hash_map.hpp` — add the memory include and default the allocator. Read the ska2 signature in `core/external/hash_map.hpp` first to get the exact Hash/Equal/Alloc parameter order and the element type ska2 allocates. Then:

```cpp
#include "./memory.hpp"
// ... existing includes ...

// flat_hash_map<K, V, Hash, Equal, Alloc> — default Hash/Equal to ska2's, Alloc to tf::allocator.
template <typename T0, typename T1,
          typename Hash = ska2::detailv3::HashOrCompare</*…match ska2 default…*/>,
          typename Equal = std::equal_to<T0>,
          typename Alloc = tf::allocator</*ska2 element type, e.g. std::pair<T0,T1>*/>>
using hash_map = ska2::flat_hash_map<T0, T1, Hash, Equal, Alloc>;
```

> The exact `Hash`/`Equal` defaults must be copied verbatim from ska2's own declaration (in `core/external/hash_map.hpp`) so behaviour is identical; only `Alloc` changes. Do the same for `hash_set.hpp`: `flat_hash_set<T, Hash, Equal, Alloc>` with `Alloc = tf::allocator<element>`.

- [ ] **Step 4: Run tests, confirm pass.**

Run: `cmake --build build-tests --target trueform_core_tests && build-tests/tests/core/trueform_core_tests "[memory]"`
Expected: PASS. Then run the full topology/cut suites (which use hash heavily) to confirm no behaviour change: `cmake --build build-tests --target trueform_cut_tests trueform_topology_tests && build-tests/tests/cut/trueform_cut_tests && build-tests/tests/topology/trueform_topology_tests`.
Expected: PASS, unchanged.

- [ ] **Step 5: Commit.**

```bash
git add include/trueform/core/hash_map.hpp include/trueform/core/hash_set.hpp tests/core/test_memory.cpp
git commit -m "feat(core): hash_map/hash_set default to tf::allocator"
```

---

## Task 3: `std::queue` → `tf::buffer` + advancing head

**Files:**
- Modify: `include/trueform/cut/arrangements/propagate_inclusion_bits.hpp` (around lines 130–161)

`std::queue<Index> q` is a strict-FIFO multi-source BFS drained once. Replace with a `tf::buffer<Index>` grown by `push_back`, read by an advancing `head` index. Same visitation order ⇒ behaviour-identical; the existing cut/csg tests cover it.

- [ ] **Step 1: Read the current loop** (`propagate_inclusion_bits.hpp:130–161`) and confirm the exact push/front/pop sites.

- [ ] **Step 2: Replace the queue.** Change the include (`#include <queue>` → remove if unused; ensure `#include "../../core/buffer.hpp"` present) and the loop:

```cpp
// was: std::queue<Index> q;
tf::buffer<Index> q;
std::size_t head = 0;
// seeding: was q.push(s);  -> q.push_back(s);
// loop:    was while (!q.empty()) { Index d = q.front(); q.pop(); ... q.push(d_other); }
while (head < q.size()) {
  const Index d = q[head++];
  // ... unchanged body ...
  // q.push(d_other);  ->  q.push_back(d_other);
}
```

Keep every other line (visited marks, XOR merges, offsets) identical. Apply the `q.push(...)` → `q.push_back(...)` rename at both push sites (seeding and the inner neighbor push).

- [ ] **Step 3: Build + run the cut/csg suites** (this function backs CSG inclusion propagation).

Run: `cmake --build build-tests --target trueform_cut_tests && build-tests/tests/cut/trueform_cut_tests`
Expected: PASS, unchanged. (If there is a csg test target, run it too.)

- [ ] **Step 4: Commit.**

```bash
git add include/trueform/cut/arrangements/propagate_inclusion_bits.hpp
git commit -m "perf(cut): replace std::queue BFS with tf::buffer + head index"
```

---

## Task 4: `finder` scratch → `tf::allocate`

**Files:**
- Modify: `include/trueform/topology/components/finder.hpp:142,147`

`std::unique_ptr<std::atomic<label_t>[]> _work_labels_ptr;` + `_work_labels_ptr.reset(new std::atomic<label_t>[size]);`. `std::atomic<label_t>` is trivially destructible. Route the allocation through `tf::allocate`, free via `tf::deallocate`.

- [ ] **Step 1: Change the member to a raw pointer + tf-managed lifetime.** Replace the `unique_ptr` member with a `unique_ptr` carrying a `tf::deallocate` deleter so the destructor still runs automatically:

```cpp
#include "../../core/memory.hpp"
// ...
struct work_labels_deleter {
  void operator()(void *p) const { tf::deallocate(p); }
};
std::unique_ptr<std::atomic<label_t>[], work_labels_deleter> _work_labels_ptr;
// allocation (was reset(new std::atomic<label_t>[size]);):
_work_labels_ptr.reset(static_cast<std::atomic<label_t> *>(
    tf::allocate(size * sizeof(std::atomic<label_t>))));
```

> `tf::allocate` returns raw bytes; `std::atomic<label_t>` is trivially default-constructible only if `label_t` is (it is, an integral). The original `new[]` value-initialized the atomics to their default (the code then fills them). Confirm the code initializes every slot before reading; it does (parallel fill follows). If any slot is read before write, add a `tf::parallel_fill`/loop to zero them — but only if the read-before-write actually exists (verify, don't assume).

- [ ] **Step 2: Build + run the topology component tests.**

Run: `cmake --build build-tests --target trueform_topology_tests && build-tests/tests/topology/trueform_topology_tests "[components]"`
Expected: PASS, unchanged. (Run the full topology suite too.)

- [ ] **Step 3: Commit.**

```bash
git add include/trueform/topology/components/finder.hpp
git commit -m "feat(topology): route finder scratch through tf::allocate"
```

---

## Task 5: Internal-only `std::vector` → `tf::vector`

**Files (internal-only sites — do NOT touch the boundary/return ones):**
- `include/trueform/cut/cut_graph.hpp:125` — `std::vector<tf::hash_map<Index, Index>>` (per-tag scratch).
- `include/trueform/csg/expression/coalesce_words.hpp:27` — `std::vector<compiled_expr::word_mask>` local `out`.
- `include/trueform/csg/expression/compile_cluster_node.hpp:42-44` — locals `leaves`, `non_leaves`.

> **Explicitly do NOT change:** `reindex/split_into_components.hpp`, `split_into_domains.hpp`, `reindex/range.hpp` returns; `csg/expression/any_of.hpp`, `all_of.hpp`, `make_children.hpp` returns; `expr.hpp`/`compiled_expr.hpp` public ctors; `core/allocate.hpp`, `core/reallocate.hpp`, `core/algorithm/generate_offset_blocks.hpp` interop overloads. These return to / accept the user's `std::vector` and stay default.

- [ ] **Step 1: For each internal site, change `std::vector<X>` → `tf::core::std_vector<X>`** and add `#include "../core/memory.hpp"` (correct relative path per file). These are locals/private scratch that never escape, so the type change is invisible to callers.

For `cut_graph.hpp:125`:
```cpp
// was: std::vector<tf::hash_map<Index, Index>> remaps(n_tags);
tf::core::std_vector<tf::hash_map<Index, Index>> remaps(n_tags);
```
For `coalesce_words.hpp:27`:
```cpp
// was: std::vector<compiled_expr::word_mask> out;
tf::core::std_vector<compiled_expr::word_mask> out;
```
For `compile_cluster_node.hpp:42-44`:
```cpp
// was: std::vector<compiled_expr::word_mask> leaves; std::vector<compiled_expr> non_leaves;
tf::core::std_vector<compiled_expr::word_mask> leaves;
tf::core::std_vector<compiled_expr> non_leaves;
```

> Verify each call site does not pass the vector into something expecting `std::vector<X>` exactly (e.g. a function param typed `const std::vector<X>&`). If it does, either change that param to a template/range or leave the site as `std::vector` (note which, and why, in the commit). `coalesce_words` takes its input by `std::vector<...> &&` — that input parameter stays `std::vector` (it accepts the caller's vector); only the local `out` becomes `tf::vector` if `out` is not returned as `std::vector`. **If `out` is returned as `std::vector<word_mask>`, leave it `std::vector` (it's boundary).** Read the return type before changing.

- [ ] **Step 2: Build the csg/cut suites.**

Run: `cmake --build build-tests --target trueform_cut_tests && build-tests/tests/cut/trueform_cut_tests`
Expected: PASS, unchanged.

- [ ] **Step 3: Commit.**

```bash
git add include/trueform/cut/cut_graph.hpp include/trueform/csg/expression/coalesce_words.hpp include/trueform/csg/expression/compile_cluster_node.hpp
git commit -m "feat: route internal scratch vectors through tf::vector"
```

---

## Task 6: `tf::buffer` storage + the 14 release frees (ATOMIC)

**This is one commit.** Changing buffer storage makes a released pointer come from `tf::allocate` (`::operator new`); a `delete[]` on it is form-mismatched UB even under the default backend. So buffer + every release-consumer move together.

**Files:**
- Modify: `include/trueform/core/buffer.hpp` (lines 94, 244, 265; release at 187).
- Modify: `vtk/src/core/make_vtk_array.cpp` (×5), `vtk/src/core/make_vtk_points.cpp`, `vtk/src/core/make_vtk_normals.cpp`, `vtk/src/core/make_vtk_cells.cpp` (×2), `vtk/include/trueform/vtk/core/make_vtk_cells.hpp` (×1).
- Modify: `python/include/trueform/python/util/make_numpy_array.hpp`, `python/include/trueform/python/spatial/gather_ids.hpp` (and the capsule it uses — `python/src/util/make_capsule.cpp` / `make_capsule.hpp`).
- Test: `tests/core/test_memory.cpp` (append a buffer round-trip).

- [ ] **Step 1: Write the failing test** (append to `tests/core/test_memory.cpp`):

```cpp
#include <trueform/core/buffer.hpp>

TEST_CASE("buffer::release yields a tf::deallocate-owned pointer", "[memory]") {
  tf::buffer<int> b;
  b.allocate(256);
  for (int i = 0; i < 256; ++i) b[i] = i;
  int *raw = b.release();          // ownership to caller
  REQUIRE(raw[255] == 255);
  tf::deallocate(raw);             // must be the matching free (ASan: no error)
}
```

- [ ] **Step 2: Run it, confirm it fails** (today `release()` returns a `new[]` pointer; `tf::deallocate` = `::operator delete` on a `new[]` pointer is mismatched — ASan will flag, or it "passes" by luck without ASan; either way this test pins the new contract).

Run: `cmake --build build-tests --target trueform_core_tests && build-tests/tests/core/trueform_core_tests "[memory]"`
Expected (with ASan): FAIL — `alloc-dealloc-mismatch` / `new[] vs operator delete`.

- [ ] **Step 3: Migrate `buffer.hpp` storage.** Add `#include "./memory.hpp"`. Define a deleter and change the storage type + the two allocation sites:

```cpp
// near the top of the class or in a detail namespace:
struct buffer_deleter {
  void operator()(void *p) const { tf::deallocate(p); }
};

// member (was: std::unique_ptr<T[]> _data = nullptr;):
std::unique_ptr<T[], buffer_deleter> _data = nullptr;

// reallocate() / allocate() sites (was: std::unique_ptr<T[]> tmp{new T[new_capacity]};):
std::unique_ptr<T[], buffer_deleter> tmp{
    static_cast<T *>(tf::allocate(new_capacity * sizeof(T)))};
```

`release()` (`return _data.release();`) is unchanged — `unique_ptr<T[], D>::release()` returns `T*`, now a `tf::allocate`'d pointer. `memcpy` lines unchanged. The `is_trivially_destructible` static_assert stays (now load-bearing for raw alloc/free).

- [ ] **Step 4: Migrate the 12 VTK frees.** In each `SetArray(ptr, n, 0, vtkAbstractArray::VTK_DATA_ARRAY_DELETE)`, change the delete method and register `tf::deallocate`:

```cpp
#include <trueform/core/memory.hpp>
// ...
arr->SetArray(ptr, static_cast<vtkIdType>(n), 0,
              vtkAbstractArray::VTK_DATA_ARRAY_USER_DEFINED);
arr->SetArrayFreeFunction(&tf::deallocate); // void(*)(void*)
```

> Confirm against the VTK version in `vtk/`: `vtkAOSDataArrayTemplate<T>::SetArrayFreeFunction(void(*)(void*))` exists (it does for VTK 9+). If `SetArrayFreeFunction` is unavailable, use `SetVoidArray` with the free callback, but `SetArrayFreeFunction` is the correct path. Apply to all 12 sites (5 in `make_vtk_array.cpp`, 2 in `make_vtk_cells.cpp`, 1 in `make_vtk_cells.hpp`, 1 in `make_vtk_points.cpp`, 1 in `make_vtk_normals.cpp`). For the `vtkPoints` path in `make_vtk_points.cpp`, the free function is set on the underlying `vtkFloatArray` before `SetData`.

- [ ] **Step 5: Migrate the 2 numpy capsule frees.** In `make_numpy_array.hpp` / `gather_ids.hpp` (and the capsule helper they use, `make_capsule`), the capsule's cleanup currently does `delete[]`. Change its body to `tf::deallocate(p)`:

```cpp
#include <trueform/core/memory.hpp>
// the capsule cleanup lambda/function body:
[](void *p) noexcept { tf::deallocate(p); }
```

> Read `python/src/util/make_capsule.cpp` / `python/include/.../make_capsule.hpp`: it likely has a templated `delete[]` cleanup. Add a path (or change the existing one) so buffers released here free via `tf::deallocate`. The per-query `new T[]` numpy results (point coords, matrices) keep their `delete[]` capsule — do NOT change those; only the two `buffer.release()` sites switch.

- [ ] **Step 6: Build + run everything under ASan.**

```bash
cmake --build build-tests --target trueform_core_tests && \
  build-tests/tests/core/trueform_core_tests "[memory]"   # buffer round-trip passes, ASan clean
# full core/cut/topology/spatial suites unchanged:
cmake --build build-tests && ctest --test-dir build-tests --output-on-failure
```
Expected: all PASS, ASan clean. Then build + smoke-test VTK and python:
```bash
# VTK: build the vtk lib + an example that goes through make_vtk_array/points, run under ASan, confirm no alloc-dealloc-mismatch on teardown.
# Python: build the extension, run the python test suite (python/tests), confirm numpy arrays from released buffers free cleanly under ASan.
```
Expected: no leaks, no `alloc-dealloc-mismatch`.

- [ ] **Step 7: Commit (atomic).**

```bash
git add include/trueform/core/buffer.hpp \
        vtk/src/core/make_vtk_array.cpp vtk/src/core/make_vtk_points.cpp \
        vtk/src/core/make_vtk_normals.cpp vtk/src/core/make_vtk_cells.cpp \
        vtk/include/trueform/vtk/core/make_vtk_cells.hpp \
        python/include/trueform/python/util/make_numpy_array.hpp \
        python/include/trueform/python/spatial/gather_ids.hpp \
        python/src/util/make_capsule.cpp python/include/trueform/python/util/make_capsule.hpp \
        tests/core/test_memory.cpp
git commit -m "feat(core): route tf::buffer through tf::allocate; release frees via tf::deallocate (VTK + numpy)"
```

---

## Task 7: Document the new contract

**Files:**
- Modify: `docs/content/cpp/2.modules/01.core.md` (the `release()` "use `delete []`" line), `docs/content/cpp/5.examples/1.mesh-assembly.md`, `CLAUDE.md:37`, `ARTICLE_DATA_PATTERNS.md:26`, `STL_ARTICLE_REFERENCE.md:247`.

- [ ] **Step 1: Update each release/ownership mention** to the `tf::deallocate` contract. The new wording, applied consistently:

> `release()` transfers ownership of the underlying array to you. It was allocated by trueform's allocator — **free it with `tf::deallocate(ptr)`**, not `delete[]`.

In `core.md`, replace "Allows you to take ownership of the underlying array (use `delete []`)" with the above. In `CLAUDE.md:37`, change the comment `// Take ownership` to `// Take ownership — free with tf::deallocate`. In `ARTICLE_DATA_PATTERNS.md` / `STL_ARTICLE_REFERENCE.md`, update the "you own it now" / "Pass to Eigen, OpenGL, or your own allocator" prose to note the free is `tf::deallocate`.

- [ ] **Step 2: Grep to confirm no `delete[]`-on-release guidance remains.**

Run: `grep -rniE 'release\(\).*delete *\[\]|delete *\[\].*release' docs CLAUDE.md *.md`
Expected: no matches (all migrated).

- [ ] **Step 3: Commit.**

```bash
git add docs/content/cpp/2.modules/01.core.md docs/content/cpp/5.examples/1.mesh-assembly.md CLAUDE.md ARTICLE_DATA_PATTERNS.md STL_ARTICLE_REFERENCE.md
git commit -m "docs: released buffer pointers free via tf::deallocate"
```

---

## Self-Review

**1. Spec coverage:**
- `core/memory.hpp` (allocate/deallocate/allocator/vector) → Task 1 ✓
- buffer → tf::allocate → Task 6 ✓
- finder scratch → Task 4 ✓
- hash_map/set default allocator → Task 2 ✓
- std::queue → buffer+head → Task 3 ✓
- internal std::vector → tf::vector; boundary stays default → Task 5 ✓ (with explicit exclusion list)
- 14 release frees → tf::deallocate → Task 6 ✓ (atomic with buffer)
- docs → Task 7 ✓
- behaviour-neutral / default backend / ASan testing → Tasks 1,6 ✓
- out-of-scope (fast backend, per-buffer templating, python per-query new[]) — not in any task ✓

**2. Placeholder scan:** No "TBD/TODO". The few "read X before editing, don't guess" notes are deliberate verification steps for ska2/VTK/numpy API exactness, not placeholders — each says precisely what to read and what to match.

**3. Type consistency:** `tf::allocate(std::size_t)->void*`, `tf::deallocate(void*)`, `tf::allocator<T>`, `tf::vector<T>`, `buffer_deleter` — used identically across Tasks 1, 4, 6. `SetArrayFreeFunction(&tf::deallocate)` matches `deallocate`'s `void(*)(void*)` (non-noexcept, by design in Task 1).

**Known verification points the executor MUST resolve (flagged inline, not guesses):** the core test target name (Task 1 pre-step), ska2's exact `Hash/Equal/Alloc` defaults + element type (Task 2), VTK `SetArrayFreeFunction` availability (Task 6 step 4), the `make_capsule` cleanup shape (Task 6 step 5), and whether `coalesce_words`'s `out` is returned (Task 5).
