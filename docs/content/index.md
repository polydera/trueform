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
`trueform` is a header-only C++ library for fast geometric processing. Compose primitives, ranges, spatial trees, and transformations inline with a clean, policy-driven API — no heavy frameworks, no copies, just your data.

#links
  :::u-button
  ---
  to: /getting-started
  size: xl
  trailing-icon: i-lucide-arrow-right
  ---
  Get started
  :::

  :::u-button
  ---
  icon: i-simple-icons-github
  color: neutral
  variant: outline
  size: xl
  to: https://github.com/xlabmedical/trueform
  target: _blank
  ---
  View on GitHub
  :::

#default
  :::prose-pre
  ---
  code: |
    // Minimal, expressive primitives
    std::vector<float> raw_points = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
    std::vector<int> triangle_indices = { 0, 1, 2 };
    auto pts = tf::make_points<3>(raw_points);
    auto triangles = tf::make_polygons(tf::make_blocked_range<3>(triangle_indices), pts);

    // Build a spatial tree and query
    tf::tree<int, float, 3> tree(triangles, tf::config_tree(4, 4));
    auto query_pt = tf::random_point<float, 3>();
    auto nearest = tf::neighbor_search(tf::make_form(tree, triangles), query_pt);
  filename: example.cpp
  ---

  ```cpp [example.cpp]
  // Minimal, expressive primitives
  std::vector<float> raw_points = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
  std::vector<int> triangle_indices = { 0, 1, 2 };
  auto pts = tf::make_points<3>(raw_points);
  auto triangles = tf::make_polygons(tf::make_blocked_range<3>(triangle_indices), pts);

  // Build a spatial tree and query
  tf::tree<int, float, 3> tree(triangles, tf::config_tree(4, 4));
  auto query_pt = tf::random_point<float, 3>();
  auto nearest = tf::neighbor_search(tf::make_form(tree, triangles), query_pt);
  ```
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
  to: /getting-started
  trailingIcon: i-lucide-arrow-right
  variant: subtle
  ---
  Explore the docs
  :::

#features
  :::u-page-feature
  ---
  icon: i-lucide-atom
  ---
  #title
  Minimal, Expressive Primitives

  #description
  Create semantically rich geometry from raw data: points, segments, polygons, AABBs, rays, and planes.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-tree-pine
  ---
  #title
  Forms and Spatial Trees

  #description
  Wrap ranges with `tf::tree` and `tf::form` for fast k-NN, intersection, and ray queries without copying your data.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-layout-grid
  ---
  #title
  Composable Policies

  #description
  Enrich primitives and ranges with `id`, `normal`, and `state` via `tag` and `zip` operations that preserve semantics through transforms.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-shapes
  ---
  #title
  Topology & Intersections

  #description
  Connectivity structures (face/vertex links, manifold edges) and exact intersections (planar arrangements, scalar slicing, mesh-mesh).
  :::

  :::u-page-feature
  ---
  icon: i-lucide-cpu
  ---
  #title
  Real-time Performance

  #description
  Header-only, parallelized, and benchmarked against VTK, CGAL, and nanoflann for speed.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-package
  ---
  #title
  Header-only Integration

  #description
  Add via CMake FetchContent. No external scene graph or heavyweight setup required.
  :::
:::

:::u-page-section{class="dark:bg-neutral-950"}
#title
Key Building Blocks

#links
  :::u-button
  ---
  color: neutral
  size: lg
  to: /getting-started/usage
  trailingIcon: i-lucide-arrow-right
  variant: subtle
  ---
  See examples
  :::

#features
  :::u-page-feature
  ---
  icon: i-lucide-axis-3d
  ---
  #title
  Primitives & Ranges

  #description
  `tf::point`, `tf::vector`, `tf::segment`, `tf::polygon` and their range counterparts with static-size propagation.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-git-branch
  ---
  #title
  Policy System

  #description
  Tag and zip metadata like IDs, normals, and composite states; preserved and transformed correctly.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-trees
  ---
  #title
  Spatial Trees & Forms

  #description
  Build `tf::tree`, wrap as `tf::form`, and run fast searches, kNN, and ray casts over static or transformed geometry.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-shield
  ---
  #title
  Robust Topology

  #description
  Face/vertex links, manifold edge link, planar embeddings, and planar graph regions for structural reasoning.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-scan-line
  ---
  #title
  Exact Intersections

  #description
  Planar arrangements, scalar-field slicing, and mesh-mesh intersection curves with edge/segment extraction.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-plug
  ---
  #title
  Header-only Integration

  #description
  C++17, Intel TBB for parallelism, and CMake FetchContent for frictionless setup.
  :::
:::

:::u-page-section{class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900"}
  :::u-page-c-t-a
  ---
  links:
    - label: Get started
      to: '/getting-started'
      trailingIcon: i-lucide-arrow-right
    - label: View on GitHub
      to: 'https://github.com/xlabmedical/trueform'
      target: _blank
      variant: subtle
      icon: i-simple-icons-github
  title: Ready to build real-time geometry?
  description: Integrate trueform into your C++ codebase with a few lines. Compose primitives, trees, and queries inline.
  class: dark:bg-neutral-950
  ---
  :::
:::
