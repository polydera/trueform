# Trueform C++ facade examples

Four standalone C++17 programs demonstrate the installed `tf::cpp` static
facade. They intentionally use only installed public headers and link only the
exported `tf::trueform_cpp` target:

- **`01_mesh_and_arrays.cpp`** holds core's own sphere buffer and a cache,
  assembles a mesh over the two, measures it, states that its points moved, and
  composes elementwise operations over the array carrier.
- **`02_async_future.cpp`** prebuilds what the worker will read, sends a mesh to
  the default asynchronous API, and verifies its exact
  `std::future<tf::polygons_buffer<...>>` result type.
- **`03_reusable_csg.cpp`** builds one immutable CSG arrangement over two
  assembled operands, extracts union, difference, and intersection meshes, and
  repairs an extracted outer shell.
- **`04_custom_async_resolvers.cpp`** sends the same async operation through a
  custom callback resolver and through a completion vendor that publishes on a
  thread of its own.

## Build from an installed package

Configure this directory directly; do not add it to a Trueform source build.
Point `CMAKE_PREFIX_PATH` at the installation prefix containing
`trueformConfig.cmake`:

```sh
cmake -S cpp/examples -B /tmp/trueform-cpp-examples \
  -DCMAKE_PREFIX_PATH=/path/to/trueform/install
cmake --build /tmp/trueform-cpp-examples --parallel
ctest --test-dir /tmp/trueform-cpp-examples --output-on-failure
```

From a copy of this directory outside the repository, replace `cpp/examples`
with `.`. A package manager or system installation that is already on CMake's
search path needs no `CMAKE_PREFIX_PATH` override. `-Dtrueform_DIR=...` may also
name the directory that directly contains `trueformConfig.cmake`.

Run an example directly to see its summary:

```sh
/tmp/trueform-cpp-examples/trueform_reusable_csg
```

## The assembly

The caller always constructs the mesh, from views into geometry it holds plus a
reference to a cache it holds plus, optionally, that instance's frame:

```cpp
tf::cpp::cache<Index, Real> cache;
tf::cpp::mesh<Index, Real> mesh{storage.faces(), storage.points(), cache};
tf::cpp::mesh<Index, Real> placed{storage.faces(), storage.points(), cache,
                                  frame};
```

Ownership is the caller's business, not this library's: core's
`tf::polygons_buffer` is the owner when a caller wants one, and VTK's arrays, a
mapping, or any other storage is as good. Every entry takes the mesh (or the
sibling carriers, `edge_mesh` and `point_cloud`), every result that is geometry
is core's own buffer, and results that are values or arrays come back as
`nd_array`. N meshes may share one geometry and one cache, each with its own
frame: a cache is built in local coordinates and the frame is a tag, so the
instances share the tree.

## The cache

The cache computes nothing the geometry does not determine — it remembers — so
dropping it loses only time. The caller states what changed:

```cpp
cache.points_changed();   // after coordinates moved
cache.faces_changed();    // after connectivity was rewired
```

Each cached structure remembers the generations it was built for, so a moved
point stales the tree and not the face membership. A mesh snapshots those
generations when it is assembled: state the change, then assemble again.

The cache is not thread safe. Before entering concurrent regions, either make
sure lazy use is safe — one filler is one filler — or prebuild what will be
read:

```cpp
tf::cpp::build_tree(mesh);
tf::cpp::build_face_membership(mesh);
tf::cpp::build_face_link(mesh);
```

Each verb builds what its structure stands on, so naming the top-level facts is
sufficient. Reads of filled state are free and unlimited.

What threads share is the geometry and the cache; a mesh is one thread's
reading of them, so each thread assembles its own:

```cpp
tf::cpp::build_tree(warm);          // on this thread, before sharing
for (int worker = 0; worker != workers; ++worker)
  results.push_back(std::async(std::launch::async, [&] {
    const tf::cpp::mesh<Index, Real> mesh{storage.faces(), storage.points(),
                                          cache};   // one reading per thread
    return tf::cpp::volume(mesh);
  }));
```

Handing one mesh VALUE to several threads is the anti-pattern: a mesh keeps
each structure the first time it is asked for, so the readers write its slots
even over a fully warm cache — a race no cache assert can see, because the
cache is never touched. Assembling is a handful of pointers and two stamps.

## Lifetime

A mesh borrows the arrays and the cache alike; both outlive it. That law is the
same synchronously and asynchronously: an async entry carries the mesh as it
stands, so the storage and the cache must outlive the future. A caller whose
storage cannot — a binding, for one — assembles with the keepalive the
constructor takes.

A `csg_graph` reads its operands for as long as it lives and holds their
readings, so their storage and caches outlive the graph. Every expression
answered against it is a read of one arrangement; `outer_shell` reads in local
coordinates, so a stored frame is neither applied nor carried onto its result.

Every async operation also accepts a resolver with a `state_type<T>` alias and
a `make_state<T>()` factory returning exactly
`std::shared_ptr<state_type<T>>`. That state supplies `result()`,
`set_value(...)`, and noexcept `set_exception(...)`. The callback example
returns `std::future<void>` while delivering the result to an application
callback on the worker thread.

The completion vendor is the runtime's one seam, set by whoever integrates this
library into a language and never visible to that language's users. It runs on a
worker thread the instant an operation finishes, and it must be cheap,
non-blocking, and must never touch a language runtime directly: it fulfills or
it posts, nothing more.
