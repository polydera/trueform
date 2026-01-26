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

#include "trueform/python/io/read_stl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace tf::py {

auto register_io_read_stl(nanobind::module_ &m) -> void {
  // Register int32 variant
  m.def("read_stl_int32",
        [](const std::string &filename) {
          return read_stl_impl<int>(filename);
        },
        nanobind::arg("filename"),
        "Read an STL file and return (faces, points) tuple with int32 indices.\n\n"
        "Parameters\n"
        "----------\n"
        "filename : str\n"
        "    Path to the STL file\n\n"
        "Returns\n"
        "-------\n"
        "faces : ndarray of shape (num_faces, 3) and dtype int32\n"
        "    Face indices into the points array\n"
        "points : ndarray of shape (num_points, 3) and dtype float32\n"
        "    3D coordinates of mesh vertices");

  // Register int64 variant
  m.def("read_stl_int64",
        [](const std::string &filename) {
          return read_stl_impl<int64_t>(filename);
        },
        nanobind::arg("filename"),
        "Read an STL file and return (faces, points) tuple with int64 indices.\n\n"
        "Parameters\n"
        "----------\n"
        "filename : str\n"
        "    Path to the STL file\n\n"
        "Returns\n"
        "-------\n"
        "faces : ndarray of shape (num_faces, 3) and dtype int64\n"
        "    Face indices into the points array\n"
        "points : ndarray of shape (num_points, 3) and dtype float32\n"
        "    3D coordinates of mesh vertices");
}

} // namespace tf::py
