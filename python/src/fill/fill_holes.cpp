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

#include "trueform/python/fill/fill_holes.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_fill_holes(nanobind::module_ &m) -> void {
  using namespace nanobind;
  // 2 index x 2 real x 2 arities, 3D meshes
  m.def(
      "fill_holes_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> vertex_offsets,
         ndarray<numpy, const int, shape<-1>> vertices,
         ndarray<numpy, const int, shape<-1>> face_offsets,
         ndarray<numpy, const int, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int, float, 3>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> triangle_offsets,
         ndarray<numpy, const int, shape<-1, 3>> triangles,
         ndarray<numpy, const int, shape<-1>> minted_offsets,
         ndarray<numpy, const float, shape<-1, 3>> minted,
         ndarray<numpy, const int, shape<-1>> plan_faces,
         ndarray<numpy, const int, shape<-1>> plan_offsets,
         ndarray<numpy, const int, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int, float, 3>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_intdynfloat3d",
      [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> vertex_offsets,
         ndarray<numpy, const int, shape<-1>> vertices,
         ndarray<numpy, const int, shape<-1>> face_offsets,
         ndarray<numpy, const int, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int, float, dynamic_size>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_intdynfloat3d",
      [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> triangle_offsets,
         ndarray<numpy, const int, shape<-1, 3>> triangles,
         ndarray<numpy, const int, shape<-1>> minted_offsets,
         ndarray<numpy, const float, shape<-1, 3>> minted,
         ndarray<numpy, const int, shape<-1>> plan_faces,
         ndarray<numpy, const int, shape<-1>> plan_offsets,
         ndarray<numpy, const int, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int, float, dynamic_size>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> vertex_offsets,
         ndarray<numpy, const int, shape<-1>> vertices,
         ndarray<numpy, const int, shape<-1>> face_offsets,
         ndarray<numpy, const int, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int, double, 3>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> triangle_offsets,
         ndarray<numpy, const int, shape<-1, 3>> triangles,
         ndarray<numpy, const int, shape<-1>> minted_offsets,
         ndarray<numpy, const double, shape<-1, 3>> minted,
         ndarray<numpy, const int, shape<-1>> plan_faces,
         ndarray<numpy, const int, shape<-1>> plan_offsets,
         ndarray<numpy, const int, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int, double, 3>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_intdyndouble3d",
      [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> vertex_offsets,
         ndarray<numpy, const int, shape<-1>> vertices,
         ndarray<numpy, const int, shape<-1>> face_offsets,
         ndarray<numpy, const int, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int, double, dynamic_size>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_intdyndouble3d",
      [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
         ndarray<numpy, const int, shape<-1>> triangle_offsets,
         ndarray<numpy, const int, shape<-1, 3>> triangles,
         ndarray<numpy, const int, shape<-1>> minted_offsets,
         ndarray<numpy, const double, shape<-1, 3>> minted,
         ndarray<numpy, const int, shape<-1>> plan_faces,
         ndarray<numpy, const int, shape<-1>> plan_offsets,
         ndarray<numpy, const int, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int, double, dynamic_size>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> vertex_offsets,
         ndarray<numpy, const int64_t, shape<-1>> vertices,
         ndarray<numpy, const int64_t, shape<-1>> face_offsets,
         ndarray<numpy, const int64_t, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int64_t, float, 3>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> triangle_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> triangles,
         ndarray<numpy, const int64_t, shape<-1>> minted_offsets,
         ndarray<numpy, const float, shape<-1, 3>> minted,
         ndarray<numpy, const int64_t, shape<-1>> plan_faces,
         ndarray<numpy, const int64_t, shape<-1>> plan_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int64_t, float, 3>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_int64dynfloat3d",
      [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> vertex_offsets,
         ndarray<numpy, const int64_t, shape<-1>> vertices,
         ndarray<numpy, const int64_t, shape<-1>> face_offsets,
         ndarray<numpy, const int64_t, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int64_t, float, dynamic_size>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_int64dynfloat3d",
      [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> triangle_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> triangles,
         ndarray<numpy, const int64_t, shape<-1>> minted_offsets,
         ndarray<numpy, const float, shape<-1, 3>> minted,
         ndarray<numpy, const int64_t, shape<-1>> plan_faces,
         ndarray<numpy, const int64_t, shape<-1>> plan_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int64_t, float, dynamic_size>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> vertex_offsets,
         ndarray<numpy, const int64_t, shape<-1>> vertices,
         ndarray<numpy, const int64_t, shape<-1>> face_offsets,
         ndarray<numpy, const int64_t, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int64_t, double, 3>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> triangle_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> triangles,
         ndarray<numpy, const int64_t, shape<-1>> minted_offsets,
         ndarray<numpy, const double, shape<-1, 3>> minted,
         ndarray<numpy, const int64_t, shape<-1>> plan_faces,
         ndarray<numpy, const int64_t, shape<-1>> plan_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int64_t, double, 3>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));

  m.def(
      "fill_holes_int64dyndouble3d",
      [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> vertex_offsets,
         ndarray<numpy, const int64_t, shape<-1>> vertices,
         ndarray<numpy, const int64_t, shape<-1>> face_offsets,
         ndarray<numpy, const int64_t, shape<-1>> faces,
         ndarray<numpy, const std::int8_t, shape<-1>> closed,
         double min_quality, bool fairing, bool split_rim) {
        return fill_holes_impl<int64_t, double, dynamic_size>(
            mesh, vertex_offsets, vertices, face_offsets, faces, closed,
            min_quality, fairing, split_rim);
      },
      arg("mesh"), arg("vertex_offsets"), arg("vertices"),
      arg("face_offsets"), arg("faces"), arg("closed"),
      arg("min_quality"), arg("fairing"), arg("split_rim"));
  m.def(
      "filled_mesh_int64dyndouble3d",
      [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
         ndarray<numpy, const int64_t, shape<-1>> triangle_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> triangles,
         ndarray<numpy, const int64_t, shape<-1>> minted_offsets,
         ndarray<numpy, const double, shape<-1, 3>> minted,
         ndarray<numpy, const int64_t, shape<-1>> plan_faces,
         ndarray<numpy, const int64_t, shape<-1>> plan_offsets,
         ndarray<numpy, const int64_t, shape<-1, 3>> plan_triangles) {
        return filled_mesh_impl<int64_t, double, dynamic_size>(
            mesh, triangle_offsets, triangles, minted_offsets, minted,
            plan_faces, plan_offsets, plan_triangles);
      },
      arg("mesh"), arg("triangle_offsets"), arg("triangles"),
      arg("minted_offsets"), arg("minted"), arg("plan_faces"),
      arg("plan_offsets"), arg("plan_triangles"));
}

} // namespace tf::py
