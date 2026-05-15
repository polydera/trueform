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

#include "./ray_cast_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_spatial_ray_cast_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::value_object<ray_cast_result_t<Real>>("RayCastResultFloat64")
      .field("hit", &ray_cast_result_t<Real>::hit)
      .field("t", &ray_cast_result_t<Real>::t)
      .field("elementId", &ray_cast_result_t<Real>::element_id);

  emscripten::value_object<ray_cast_prim_batch_result_t<Real>>(
      "RayCastPrimBatchResultFloat64")
      .field("hits", &ray_cast_prim_batch_result_t<Real>::hits)
      .field("ts", &ray_cast_prim_batch_result_t<Real>::ts);

  emscripten::value_object<ray_cast_form_batch_result_t<Real>>(
      "RayCastFormBatchResultFloat64")
      .field("hits", &ray_cast_form_batch_result_t<Real>::hits)
      .field("ts", &ray_cast_form_batch_result_t<Real>::ts)
      .field("elementIds", &ray_cast_form_batch_result_t<Real>::element_ids);

  emscripten::function("ray_cast_p_float64", &sync_ray_cast_p<Real>);
  emscripten::function("ray_cast_p_rc_float64", &sync_ray_cast_p_rc<Real>);
  emscripten::function("ray_cast_f_float64", &sync_ray_cast_f<Real>);
  emscripten::function("ray_cast_f_rc_float64", &sync_ray_cast_f_rc<Real>);
  emscripten::function("ray_cast_f_pc_float64", &sync_ray_cast_f_pc<Real>);
  emscripten::function("ray_cast_f_pc_rc_float64", &sync_ray_cast_f_pc_rc<Real>);
  emscripten::function("dispatch_ray_cast_p_float64", &async_ray_cast_p<Real>);
  emscripten::function("dispatch_ray_cast_p_rc_float64",
                       &async_ray_cast_p_rc<Real>);
  emscripten::function("dispatch_ray_cast_f_float64", &async_ray_cast_f<Real>);
  emscripten::function("dispatch_ray_cast_f_rc_float64",
                       &async_ray_cast_f_rc<Real>);
  emscripten::function("dispatch_ray_cast_f_pc_float64",
                       &async_ray_cast_f_pc<Real>);
  emscripten::function("dispatch_ray_cast_f_pc_rc_float64",
                       &async_ray_cast_f_pc_rc<Real>);
}
