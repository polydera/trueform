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

#include "./io_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_io_float64) {
  using Real = double;
  using namespace tf::ts::io;

  // Result types
  emscripten::value_object<mesh_data_result_t<Real>>("MeshDataResultFloat64")
      .field("faces", &mesh_data_result_t<Real>::faces)
      .field("points", &mesh_data_result_t<Real>::points);

  // OBJ read (float64)
  emscripten::function("read_obj_float64", &sync_read_obj<Real>);
  emscripten::function("dispatch_read_obj_float64", &async_read_obj<Real>);
  emscripten::function("read_obj_buffer_float64",
                       &sync_read_obj_buffer<Real>);
  emscripten::function("dispatch_read_obj_buffer_float64",
                       &async_read_obj_buffer<Real>);
  emscripten::function("read_obj_buffer_data_float64",
                       &sync_read_obj_buffer_data<Real>);
  emscripten::function("dispatch_read_obj_buffer_data_float64",
                       &async_read_obj_buffer_data<Real>);

  // STL write (float64 input; output stays float32 per STL spec)
  emscripten::function("write_stl_buffer_float64",
                       &sync_write_stl_buffer<Real>);
  emscripten::function("dispatch_write_stl_buffer_float64",
                       &async_write_stl_buffer<Real>);

  // OBJ write (float64 input; output uses %.17g precision)
  emscripten::function("write_obj_buffer_float64",
                       &sync_write_obj_buffer<Real>);
  emscripten::function("dispatch_write_obj_buffer_float64",
                       &async_write_obj_buffer<Real>);
}
