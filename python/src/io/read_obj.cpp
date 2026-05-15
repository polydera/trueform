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

namespace {

constexpr const char *kFixedDoc =
    "Read an OBJ file and return (faces, points) tuple.\n\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to the OBJ file\n";

constexpr const char *kDynamicDoc =
    "Read an OBJ file with dynamic polygon sizes.\n\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to the OBJ file\n";

} // namespace

auto register_io_read_obj(nanobind::module_ &m) -> void {
  // ---- Fixed-size (triangles and quads), float32 ----
  m.def(
      "read_obj_int3float3d",
      [](const std::string &filename) {
        return read_obj_impl<int, float, 3>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  m.def(
      "read_obj_int4float3d",
      [](const std::string &filename) {
        return read_obj_impl<int, float, 4>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  m.def(
      "read_obj_int643float3d",
      [](const std::string &filename) {
        return read_obj_impl<int64_t, float, 3>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  m.def(
      "read_obj_int644float3d",
      [](const std::string &filename) {
        return read_obj_impl<int64_t, float, 4>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  // ---- Fixed-size (triangles and quads), float64 ----
  m.def(
      "read_obj_int3double3d",
      [](const std::string &filename) {
        return read_obj_impl<int, double, 3>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  m.def(
      "read_obj_int4double3d",
      [](const std::string &filename) {
        return read_obj_impl<int, double, 4>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  m.def(
      "read_obj_int643double3d",
      [](const std::string &filename) {
        return read_obj_impl<int64_t, double, 3>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  m.def(
      "read_obj_int644double3d",
      [](const std::string &filename) {
        return read_obj_impl<int64_t, double, 4>(filename);
      },
      nanobind::arg("filename"), kFixedDoc);

  // ---- Dynamic polygon sizes, float32 ----
  m.def(
      "read_obj_intdynfloat3d",
      [](const std::string &filename) {
        return read_obj_dynamic_impl<int, float>(filename);
      },
      nanobind::arg("filename"), kDynamicDoc);

  m.def(
      "read_obj_int64dynfloat3d",
      [](const std::string &filename) {
        return read_obj_dynamic_impl<int64_t, float>(filename);
      },
      nanobind::arg("filename"), kDynamicDoc);

  // ---- Dynamic polygon sizes, float64 ----
  m.def(
      "read_obj_intdyndouble3d",
      [](const std::string &filename) {
        return read_obj_dynamic_impl<int, double>(filename);
      },
      nanobind::arg("filename"), kDynamicDoc);

  m.def(
      "read_obj_int64dyndouble3d",
      [](const std::string &filename) {
        return read_obj_dynamic_impl<int64_t, double>(filename);
      },
      nanobind::arg("filename"), kDynamicDoc);
}

} // namespace tf::py
