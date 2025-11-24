---
seo:
  title: trueform — Real-time geometric processing
  description: A C++ header-only library for fast geometric queries, forms and topology built on composable range-based policies.
---

::u-page-hero{class="dark:bg-gradient-to-b from-neutral-900 to-neutral-950"}
---
orientation: horizontal
---
#top
:hero-background

#title
Real-time [geometric processing]{.text-primary} for C++.

#description
`trueform` is a header-only C++ library for real-time geometric processing. Build spatial queries, mesh intersections, boolean operations, and topology analysis directly on your data with composable, zero-copy views — no heavy frameworks, no architectural changes, just semantic wrappers over your existing buffers.

#links
  :::u-button
  ---
  to: /cpp/getting-started
  size: xl
  variant: subtle
  trailing-icon: i-vscode-icons:file-type-cpp
  ---
  Get started
  :::

  :::u-button
  ---
  to: /py/getting-started
  size: xl
  variant: subtle
  trailing-icon: i-vscode-icons:file-type-python
  ---
  Get started
  :::

  :::u-button
  ---
  icon: i-lucide-play
  color: primary
  variant: solid
  size: xl
  to: /live-examples/boolean
  class: animate-pulse
  ---
  Try it live
  :::

#default
  ::chart-carousel
  :::
::

:::u-page-section{class="dark:bg-neutral-950"}
#title
Why trueform

#links
  :::u-button
  ---
  color: neutral
  size: lg
  to: /cpp/modules/core
  trailingIcon: i-lucide-arrow-right
  variant: subtle
  ---
  Explore the docs
  :::

  :::u-button
  ---
  color: neutral
  size: lg
  to: /cpp/benchmarks
  trailingIcon: i-lucide-chart-line
  variant: subtle
  ---
  See benchmarks
  :::

#features
  :::u-page-feature
  ---
  icon: i-lucide-atom
  ---
  #title
  Zero-Copy Views

  #description
  Work directly on your data layout with semantic geometric wrappers. Enrich primitives with `id`, `normal`, and `state` via composable `tag` and `zip` operations.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-tree-pine
  ---
  #title
  Spatial Acceleration

  #description
  Build `tf::tree` for k-NN, neighbor search, ray casting, and broad-phase queries. Wrap with `tf::form` to add transformations without copying data.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-shapes
  ---
  #title
  Topology & Intersections

  #description
  Connectivity structures, boundary detection, path finding. Mesh-mesh curves, self-intersections, scalar field isocontours with topological classification.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-scissors
  ---
  #title
  Cut & Boolean Operations

  #description
  Embed intersection curves as edges via face splitting. Boolean operations (union, intersection, difference) and planar arrangements for 2D subdivision.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-database
  ---
  #title
  Data Management

  #description
  Flat buffers with direct memory access. Reindexing and filtering with automatic referential integrity. Cleaning operations for duplicates and degenerates.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-gauge
  ---
  #title
  Real-time Performance

  #description
  Parallel algorithms built on Intel TBB with optimized memory layouts. Benchmarked against VTK, CGAL, nanoflann, and IGL.
  :::
:::

:::u-page-section{class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900"}
  :::u-page-c-t-a
  ---
  links:
    - label: Read the tutorial
      to: '/cpp/modules/core'
      trailingIcon: i-lucide-arrow-right
    - label: View benchmarks
      to: '/cpp/benchmarks'
      variant: subtle
      trailingIcon: i-lucide-chart-line
  title: Ready to build real-time geometry?
  description: Integrate trueform into your C++ codebase with CMake FetchContent. Process meshes, compute intersections, and perform spatial queries directly on your data.
  class: dark:bg-neutral-950
  ---
  :::
:::
