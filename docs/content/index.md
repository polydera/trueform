---
seo:
  title: trueform — Real-time geometric processing
  description: Geometry library for real-time spatial queries, mesh booleans, and topology. C++ header-only with Python bindings.
---

::u-page-hero{class="dark:bg-gradient-to-b from-neutral-900 to-neutral-950"}
---
orientation: horizontal
---
#top
:hero-background

#title
Real-time [geometric processing]{.text-primary}

#description
Spatial queries, mesh booleans, isocontours, topology — at interactive speed on million-polygon meshes. Robust on real-world inputs: non-manifold flaps, inconsistent geometry, the artifacts that pipelines accumulate. Algorithms with formal guarantees. C++ header-only; Python with NumPy in and out.

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
  icon: i-lucide-play
  color: primary
  variant: solid
  size: lg
  to: /live-examples/boolean
  ---
  Try it live
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
  icon: i-lucide-shield-check
  ---
  #title
  Robust by Design

  #description
  Production-tested on meshes with non-manifold flaps, inconsistent geometry, and accumulated pipeline artifacts. Algorithms with formal correctness guarantees.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-gauge
  ---
  #title
  Real-time Performance

  #description
  Interactive speed on million-polygon meshes. Parallel algorithms benchmarked against VTK, CGAL, libigl, FCL, and nanoflann.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-atom
  ---
  #title
  Zero-Copy Views

  #description
  Wrap your existing data with geometric meaning. No copies, no conversions, no new types to learn. Your buffers, enriched with spatial semantics.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-tree-pine
  ---
  #title
  Spatial Acceleration

  #description
  Fast spatial queries on point clouds, curves, and meshes. k-NN, closest points, ray casting, collision detection. Transform geometry without rebuilding acceleration structures.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-shapes
  ---
  #title
  Topology & Intersections

  #description
  Understand mesh structure—connectivity, boundaries, connected components. Find where meshes meet: intersection curves, self-intersections, isocontours.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-scissors
  ---
  #title
  Cut & Boolean Operations

  #description
  Combine and cut meshes with union, intersection, difference. Commutative correctness: defer mesh cleanup to the final step without corrupting results.
  :::
:::

:::u-page-section{class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900"}
  :::u-page-c-t-a
  ---
  links:
    - label: C++ with CMake
      to: '/cpp/getting-started/installation'
      trailingIcon: i-lucide-arrow-right
    - label: Python with pip
      to: '/py/getting-started/installation'
      variant: subtle
      trailingIcon: i-lucide-arrow-right
  title: Ready to build?
  description: C++ via CMake FetchContent. Python via pip install.
  class: dark:bg-neutral-950
  ---
  :::
:::
