---
seo:
  title: trueform — Real-time geometric processing
  description: Geometry library for real-time mesh booleans, registration, remeshing and queries. C++ header-only with Python bindings.
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
Mesh booleans, registration, remeshing and queries — at interactive speed on million-polygon meshes. Robust to non-manifold flaps, inconsistent winding, and pipeline artifacts. C++ header-only; Python with NumPy in, NumPy out.

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
  ::
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
  class: animate-pulse
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
  icon: i-lucide-atom
  ---
  #title
  Easy to Use

  #description
  No boilerplate. Ranges wrap your data with geometric meaning. Tag structures onto them for compile-time algorithm selection.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-shield-check
  ---
  #title
  Robust by Design

  #description
  Production-tested on meshes with non-manifold flaps, inconsistent geometry, and accumulated pipeline artifacts.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-gauge
  ---
  #title
  Real-time Performance

  #description
  Interactive speed on million-polygon meshes. Algorithms benchmarked against VTK, CGAL, libigl, Coal, FCL, and nanoflann.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-tree-pine
  ---
  #title
  Queries & Topology

  #description
  Spatial trees for collision detection, distance queries, ray casting, and k-NN on moving geometry. Connectivity analysis, boundaries, and connected components.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-scissors
  ---
  #title
  Booleans & Intersections

  #description
  Union, intersection, and difference on closed meshes. Intersection curves, isocontours, and isobands. Commutative correctness: chain operations freely.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-triangle
  ---
  #title
  Remeshing & Registration

  #description
  Decimation and isotropic remeshing for mesh simplification and quality improvement. Point cloud alignment with ICP, OBB fitting, and rigid registration.
  :::
:::

:::u-page-section{class="dark:bg-neutral-950"}
#title
Integrations

#default
::card-group
  :::card
  ---
  icon: i-lucide-layers
  title: VTK
  to: /cpp/vtk
  ---
  Bring trueform performance to VTK applications. Filters and functions that integrate with VTK pipelines.
  :::

  :::card
  ---
  icon: i-vscode-icons:file-type-python
  title: Python
  to: /py/getting-started
  ---
  Real-time geometric processing in your Python workflow. NumPy in, NumPy out.
  :::

  :::card
  ---
  icon: i-vscode-icons:file-type-blender
  title: Blender
  to: /py/blender
  ---
  Bring trueform performance to Blender add-ons. Cached meshes with automatic updates for live preview.
  :::
::
:::

:::u-page-section{class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900"}
  :::u-page-c-t-a
  ---
  links:
    - label: C++
      to: '/cpp/getting-started'
      icon: i-vscode-icons:file-type-cpp
      trailingIcon: i-lucide-arrow-right
    - label: Python
      to: '/py/getting-started'
      icon: i-vscode-icons:file-type-python
      variant: subtle
      trailingIcon: i-lucide-arrow-right
  title: Start now
  description: From pip install to mesh booleans — in minutes.
  class: dark:bg-neutral-950
  ---
  :::
:::
