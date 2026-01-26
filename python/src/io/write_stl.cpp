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

#include "trueform/python/io/write_stl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace tf::py {

auto register_io_write_stl(nanobind::module_ &m) -> void {
  // Register int32 variant
  m.def("write_stl_int32",
        [](nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>
               faces_array,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               points_array,
           std::optional<nanobind::ndarray<nanobind::numpy, float,
                                           nanobind::shape<4, 4>>>
               transformation_opt,
           const std::string &filename) {
          return write_stl_impl<int>(faces_array, points_array,
                                     transformation_opt, filename);
        },
        nanobind::arg("faces"), nanobind::arg("points"),
        nanobind::arg("transformation").none() = nanobind::none(),
        nanobind::arg("filename"),
        "Write a triangular mesh to an STL file with int32 indices.\n\n"
        "Parameters\n"
        "----------\n"
        "faces : ndarray of shape (num_faces, 3) and dtype int32\n"
        "    Face indices into the points array\n"
        "points : ndarray of shape (num_points, 3) and dtype float32\n"
        "    3D coordinates of mesh vertices\n"
        "transformation : ndarray of shape (4, 4) and dtype float32, optional\n"
        "    Transformation matrix to apply before writing\n"
        "filename : str\n"
        "    Path to output STL file (.stl extension added if not present)\n\n"
        "Returns\n"
        "-------\n"
        "success : bool\n"
        "    True if write succeeded, False otherwise");

  // Register int64 variant
  m.def("write_stl_int64",
        [](nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>
               faces_array,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               points_array,
           std::optional<nanobind::ndarray<nanobind::numpy, float,
                                           nanobind::shape<4, 4>>>
               transformation_opt,
           const std::string &filename) {
          return write_stl_impl<int64_t>(faces_array, points_array,
                                         transformation_opt, filename);
        },
        nanobind::arg("faces"), nanobind::arg("points"),
        nanobind::arg("transformation").none() = nanobind::none(),
        nanobind::arg("filename"),
        "Write a triangular mesh to an STL file with int64 indices.\n\n"
        "Parameters\n"
        "----------\n"
        "faces : ndarray of shape (num_faces, 3) and dtype int64\n"
        "    Face indices into the points array\n"
        "points : ndarray of shape (num_points, 3) and dtype float32\n"
        "    3D coordinates of mesh vertices\n"
        "transformation : ndarray of shape (4, 4) and dtype float32, optional\n"
        "    Transformation matrix to apply before writing\n"
        "filename : str\n"
        "    Path to output STL file (.stl extension added if not present)\n\n"
        "Returns\n"
        "-------\n"
        "success : bool\n"
        "    True if write succeeded, False otherwise");
}

} // namespace tf::py
