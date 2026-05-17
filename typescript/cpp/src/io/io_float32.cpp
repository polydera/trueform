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

EMSCRIPTEN_BINDINGS(trueform_io_float32) {
  using Real = float;
  using namespace tf::ts::io;

  // Result types
  emscripten::value_object<mesh_data_result_t<Real>>("MeshDataResultFloat32")
      .field("faces", &mesh_data_result_t<Real>::faces)
      .field("points", &mesh_data_result_t<Real>::points);

  // STL read (float32-only by format spec)
  emscripten::function("read_stl", &sync_read_stl);
  emscripten::function("dispatch_read_stl", &async_read_stl);
  emscripten::function("read_stl_buffer", &sync_read_stl_buffer);
  emscripten::function("dispatch_read_stl_buffer", &async_read_stl_buffer);

  // OBJ read (float32)
  emscripten::function("read_obj_float32", &sync_read_obj<Real>);
  emscripten::function("dispatch_read_obj_float32", &async_read_obj<Real>);
  emscripten::function("read_obj_buffer_float32",
                       &sync_read_obj_buffer<Real>);
  emscripten::function("dispatch_read_obj_buffer_float32",
                       &async_read_obj_buffer<Real>);
  emscripten::function("read_obj_buffer_data_float32",
                       &sync_read_obj_buffer_data<Real>);
  emscripten::function("dispatch_read_obj_buffer_data_float32",
                       &async_read_obj_buffer_data<Real>);

  // STL write (float32 input)
  emscripten::function("write_stl_buffer", &sync_write_stl_buffer<Real>);
  emscripten::function("dispatch_write_stl_buffer",
                       &async_write_stl_buffer<Real>);

  // OBJ write (float32 input)
  emscripten::function("write_obj_buffer_float32",
                       &sync_write_obj_buffer<Real>);
  emscripten::function("dispatch_write_obj_buffer_float32",
                       &async_write_obj_buffer<Real>);
}
