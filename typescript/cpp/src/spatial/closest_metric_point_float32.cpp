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

#include "./closest_metric_point_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_spatial_closest_metric_point_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<metric_point_result_t<Real>>(
      "MetricPointResultFloat32")
      .field("point", &metric_point_result_t<Real>::point)
      .field("distance2", &metric_point_result_t<Real>::distance2);

  emscripten::value_object<metric_point_batch_result_t<Real>>(
      "MetricPointBatchResultFloat32")
      .field("points", &metric_point_batch_result_t<Real>::points)
      .field("distances", &metric_point_batch_result_t<Real>::distances);

  emscripten::function("closest_metric_point_float32",
                       &sync_closest_metric_point<Real>);
  emscripten::function("dispatch_closest_metric_point_float32",
                       &async_closest_metric_point<Real>);
}
