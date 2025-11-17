---
title: Benchmarks
description: Comprehensive performance comparisons between trueform and industry-standard libraries across various computational geometry tasks.
navigation:
  icon: i-lucide-chart-bar-stacked
---

## Overview

The benchmarks below compare <span class="text-primary">**trueform**</span> against established libraries including VTK, CGAL, nanoflann, and IGL across different computational geometry operations. All benchmarks were executed on an Intel i7-9750H with parallel execution enabled for fair comparison.

For implementation details and source code, see the [comparison examples](/cpp/examples/comparisons).


## Spatial Trees and Queries

Comparing spatial acceleration structures and proximity queries against **nanoflann** and **CGAL**. See [k-NN queries](/cpp/examples/comparisons#k-nn-query-performance) and [collecting intersecting primitives](/cpp/examples/comparisons#collecting-intersecting-primitives) examples.

:ChartsTreeReconstruction{class="my-12"}
::

::ChartsKnn{class="my-12"}
::

::ChartsTreeConstructionCGAL{class="my-12"}
::

## Feature Edges

Extracting boundary and non-manifold edges against **VTK**. Demonstrates efficient topology analysis using trueform's manifold edge link data structure. See [feature edges example](/cpp/examples/comparisons#feature-edges).

::ChartsFeatureEdgesBoundary{class="my-12"}
::

::ChartsFeatureEdgesNonManifold{class="my-12"}
::

## Intersection Curves

Computing intersection curves between two meshes against **CGAL** and **VTK**. Performance advantage grows with mesh size, demonstrating superior scaling characteristics in spatial indexing and parallel intersection algorithms. See [intersection curves (CGAL)](/cpp/examples/comparisons#intersection-curves) and [intersection curves (VTK)](/cpp/examples/comparisons#intersection-curves-1) examples.

::ChartsIntersectionCurveComparison{class="my-12"}
::

## Isocontours

Extracting isocontours from scalar fields on meshes against **VTK**. Single isocontour extraction shows 10x to 27x speedup, while multi-contour extraction demonstrates superlinear speedup growth (10x to 170x) as trueform's `build_many` amortizes computation across threshold levels.

::ChartsIsoContours{class="my-12"}
::

::ChartsIsoContoursMany{class="my-12"}
::

## Planar Arrangements

Computing 2D planar arrangements from unordered line segments against **CGAL**. Demonstrates the most dramatic performance difference, with speedup growing from 1.3x at 200 edges to 2273x at 40k edges. CGAL exhibits superlinear complexity growth while trueform maintains near-linear scaling. See [planar arrangements example](/cpp/examples/comparisons#planar-arrangements).

::ChartsPlanarArrangements{class="my-16"}
::

## Mesh Booleans

Computing boolean operations (union, intersection, difference) between two meshes against **CGAL**. Speedup ranges from 30x to 80x depending on mesh size and operation complexity. See [mesh booleans example](/cpp/examples/comparisons#mesh-booleans).


::ChartsBoolean{class="my-12"}
::

::ChartsBooleanBars{class="my-12"}
::
