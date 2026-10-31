/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

#include "./fill_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_fill_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::value_object<hole_fill_t<Real>>("HoleFillFloat64")
      .field("groupOffsets", &hole_fill_t<Real>::group_offsets)
      .field("triangleOffsets", &hole_fill_t<Real>::triangle_offsets)
      .field("triangles", &hole_fill_t<Real>::triangles)
      .field("mintedOffsets", &hole_fill_t<Real>::minted_offsets)
      .field("mintedPoints", &hole_fill_t<Real>::minted_points)
      .field("status", &hole_fill_t<Real>::status)
      .field("charted", &hole_fill_t<Real>::charted)
      .field("refined", &hole_fill_t<Real>::refined)
      .field("faired", &hole_fill_t<Real>::faired)
      .field("seamMaxAngle", &hole_fill_t<Real>::seam_max_angle)
      .field("offending", &hole_fill_t<Real>::offending)
      .field("splits", &hole_fill_t<Real>::splits)
      .field("splitParameters", &hole_fill_t<Real>::split_parameters)
      .field("planFaces", &hole_fill_t<Real>::plan_faces)
      .field("planOffsets", &hole_fill_t<Real>::plan_offsets)
      .field("planTriangles", &hole_fill_t<Real>::plan_triangles);

  emscripten::function("fill_holes_float64", &sync_fill_holes<Real>);
  emscripten::function("dispatch_fill_holes_float64", &async_fill_holes<Real>);
  emscripten::function("filled_mesh_float64", &sync_filled_mesh<Real>);
  emscripten::function("dispatch_filled_mesh_float64", &async_filled_mesh<Real>);
}
