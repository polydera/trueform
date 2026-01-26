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

#include "trueform/python/io/write_obj.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace tf::py {

auto register_io_write_obj(nanobind::module_ &m) -> void {
  // int32, float32, triangles
  m.def(
      "write_obj_int3float3d",
      [](nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int, float, 3>(faces_array, points_array,
                                             transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a triangular mesh to an OBJ file with int32 indices and float32 "
      "points.");

  // int32, float32, quads
  m.def(
      "write_obj_int4float3d",
      [](nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 4>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int, float, 4>(faces_array, points_array,
                                             transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a quad mesh to an OBJ file with int32 indices and float32 "
      "points.");

  // int64, float32, triangles
  m.def(
      "write_obj_int643float3d",
      [](nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int64_t, float, 3>(faces_array, points_array,
                                                 transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a triangular mesh to an OBJ file with int64 indices and float32 "
      "points.");

  // int64, float32, quads
  m.def(
      "write_obj_int644float3d",
      [](nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 4>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int64_t, float, 4>(faces_array, points_array,
                                                 transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a quad mesh to an OBJ file with int64 indices and float32 "
      "points.");

  // int32, double, triangles
  m.def(
      "write_obj_int3double3d",
      [](nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int, double, 3>(faces_array, points_array,
                                              transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a triangular mesh to an OBJ file with int32 indices and float64 "
      "points.");

  // int32, double, quads
  m.def(
      "write_obj_int4double3d",
      [](nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 4>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int, double, 4>(faces_array, points_array,
                                              transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a quad mesh to an OBJ file with int32 indices and float64 "
      "points.");

  // int64, double, triangles
  m.def(
      "write_obj_int643double3d",
      [](nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int64_t, double, 3>(faces_array, points_array,
                                                  transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a triangular mesh to an OBJ file with int64 indices and float64 "
      "points.");

  // int64, double, quads
  m.def(
      "write_obj_int644double3d",
      [](nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 4>>
             faces_array,
         nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
             points_array,
         std::optional<
             nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
             transformation_opt,
         const std::string &filename) {
        return write_obj_impl<int64_t, double, 4>(faces_array, points_array,
                                                  transformation_opt, filename);
      },
      nanobind::arg("faces"), nanobind::arg("points"),
      nanobind::arg("transformation").none() = nanobind::none(),
      nanobind::arg("filename"),
      "Write a quad mesh to an OBJ file with int64 indices and float64 "
      "points.");
}

} // namespace tf::py
