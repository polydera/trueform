/*
 * Copyright (c) 2026 XLAB
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

#include "./surface_fields_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_surface_fields_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<dirichlet_field_t<Real>>("DirichletFieldFloat32")
      .field("field", &dirichlet_field_t<Real>::field)
      .field("status", &dirichlet_field_t<Real>::status);

  emscripten::function("geodesic_distances_float32",
                       &sync_geodesic_distances<Real>);
  emscripten::function("dirichlet_field_float32", &sync_dirichlet_field<Real>);
  emscripten::function("dirichlet_field_cotangent_float32",
                       &sync_dirichlet_field_cotangent<Real>);

  emscripten::function("dispatch_geodesic_distances_float32",
                       &async_geodesic_distances<Real>);
  emscripten::function("dispatch_dirichlet_field_float32",
                       &async_dirichlet_field<Real>);
  emscripten::function("dispatch_dirichlet_field_cotangent_float32",
                       &async_dirichlet_field_cotangent<Real>);
}
