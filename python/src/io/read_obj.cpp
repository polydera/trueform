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

#include "trueform/python/io/read_obj.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace tf::py {

auto register_io_read_obj(nanobind::module_ &m) -> void {
  // Register int32 triangles variant
  m.def(
      "read_obj_int3",
      [](const std::string &filename) {
        return read_obj_impl<int, 3>(filename);
      },
      nanobind::arg("filename"),
      "Read an OBJ file and return (faces, points) tuple with int32 indices "
      "and triangular faces.\n\n"
      "Parameters\n"
      "----------\n"
      "filename : str\n"
      "    Path to the OBJ file\n\n"
      "Returns\n"
      "-------\n"
      "faces : ndarray of shape (num_faces, 3) and dtype int32\n"
      "    Face indices into the points array\n"
      "points : ndarray of shape (num_points, 3) and dtype float32\n"
      "    3D coordinates of mesh vertices");

  // Register int32 quads variant
  m.def(
      "read_obj_int4",
      [](const std::string &filename) {
        return read_obj_impl<int, 4>(filename);
      },
      nanobind::arg("filename"),
      "Read an OBJ file and return (faces, points) tuple with int32 indices "
      "and quad faces.\n\n"
      "Parameters\n"
      "----------\n"
      "filename : str\n"
      "    Path to the OBJ file\n\n"
      "Returns\n"
      "-------\n"
      "faces : ndarray of shape (num_faces, 4) and dtype int32\n"
      "    Face indices into the points array\n"
      "points : ndarray of shape (num_points, 3) and dtype float32\n"
      "    3D coordinates of mesh vertices");

  // Register int64 triangles variant
  m.def(
      "read_obj_int643",
      [](const std::string &filename) {
        return read_obj_impl<int64_t, 3>(filename);
      },
      nanobind::arg("filename"),
      "Read an OBJ file and return (faces, points) tuple with int64 indices "
      "and triangular faces.\n\n"
      "Parameters\n"
      "----------\n"
      "filename : str\n"
      "    Path to the OBJ file\n\n"
      "Returns\n"
      "-------\n"
      "faces : ndarray of shape (num_faces, 3) and dtype int64\n"
      "    Face indices into the points array\n"
      "points : ndarray of shape (num_points, 3) and dtype float32\n"
      "    3D coordinates of mesh vertices");

  // Register int64 quads variant
  m.def(
      "read_obj_int644",
      [](const std::string &filename) {
        return read_obj_impl<int64_t, 4>(filename);
      },
      nanobind::arg("filename"),
      "Read an OBJ file and return (faces, points) tuple with int64 indices "
      "and quad faces.\n\n"
      "Parameters\n"
      "----------\n"
      "filename : str\n"
      "    Path to the OBJ file\n\n"
      "Returns\n"
      "-------\n"
      "faces : ndarray of shape (num_faces, 4) and dtype int64\n"
      "    Face indices into the points array\n"
      "points : ndarray of shape (num_points, 3) and dtype float32\n"
      "    3D coordinates of mesh vertices");
}

} // namespace tf::py
