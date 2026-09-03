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
#include "trueform/python/topology/euler_characteristic_impl.hpp"

namespace tf::py {

auto register_topology_euler_characteristic(nanobind::module_ &m) -> void {
  using namespace nanobind;
  m.def(
      "euler_characteristic_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &mesh) {
        return impl::euler_characteristic_impl<int, 3, float, 3>(mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &mesh) {
        return impl::euler_characteristic_impl<int, 3, double, 3>(mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_int643float3d",
      [](mesh_wrapper<std::int64_t, float, 3, 3> &mesh) {
        return impl::euler_characteristic_impl<std::int64_t, 3, float, 3>(
            mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_int643double3d",
      [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh) {
        return impl::euler_characteristic_impl<std::int64_t, 3, double, 3>(
            mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_intdynfloat3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh) {
        return impl::euler_characteristic_impl<int, tf::dynamic_size, float,
                                               3>(mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_intdyndouble3d",
      [](mesh_wrapper<int, double, tf::dynamic_size, 3> &mesh) {
        return impl::euler_characteristic_impl<int, tf::dynamic_size, double,
                                               3>(mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_int64dynfloat3d",
      [](mesh_wrapper<std::int64_t, float, tf::dynamic_size, 3> &mesh) {
        return impl::euler_characteristic_impl<std::int64_t, tf::dynamic_size,
                                               float, 3>(mesh);
      },
      arg("mesh"));
  m.def(
      "euler_characteristic_int64dyndouble3d",
      [](mesh_wrapper<std::int64_t, double, tf::dynamic_size, 3> &mesh) {
        return impl::euler_characteristic_impl<std::int64_t, tf::dynamic_size,
                                               double, 3>(mesh);
      },
      arg("mesh"));
}

} // namespace tf::py
