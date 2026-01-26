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

#include "trueform/python/spatial/point_cloud.hpp"

namespace tf::py {

// Macro to define bindings for point_cloud_wrapper
#define REGISTER_POINT_CLOUD_WRAPPER(RealT, Dims, Name)                        \
  nanobind::class_<point_cloud_wrapper<RealT, Dims>>(m, Name)                  \
      .def(nanobind::init<                                                     \
           nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>()) \
      .def("build_tree", &point_cloud_wrapper<RealT, Dims>::build_tree)        \
      .def("has_tree", &point_cloud_wrapper<RealT, Dims>::has_tree)            \
      .def("size", &point_cloud_wrapper<RealT, Dims>::size)                    \
      .def("dims", &point_cloud_wrapper<RealT, Dims>::dims)                    \
      .def("points_array", &point_cloud_wrapper<RealT, Dims>::points_array)    \
      .def("set_points_array",                                                 \
           &point_cloud_wrapper<RealT, Dims>::set_points_array)                \
      .def("has_transformation",                                               \
           &point_cloud_wrapper<RealT, Dims>::has_transformation)              \
      .def("transformation",                                                   \
           &point_cloud_wrapper<RealT, Dims>::transformation)                  \
      .def("set_transformation",                                               \
           &point_cloud_wrapper<RealT, Dims>::set_transformation)              \
      .def("clear_transformation",                                             \
           &point_cloud_wrapper<RealT, Dims>::clear_transformation)            \
      .def("mark_modified",                                                    \
           &point_cloud_wrapper<RealT, Dims>::mark_modified)                   \
      .def("shared_view", &point_cloud_wrapper<RealT, Dims>::shared_view)

auto register_point_cloud(nanobind::module_ &m) -> void {
  REGISTER_POINT_CLOUD_WRAPPER(float, 2, "PointCloudWrapperFloat2D");
  REGISTER_POINT_CLOUD_WRAPPER(float, 3, "PointCloudWrapperFloat3D");
  REGISTER_POINT_CLOUD_WRAPPER(double, 2, "PointCloudWrapperDouble2D");
  REGISTER_POINT_CLOUD_WRAPPER(double, 3, "PointCloudWrapperDouble3D");
}

#undef REGISTER_POINT_CLOUD_WRAPPER

// Explicit template instantiations for point_cloud_data_wrapper
template class point_cloud_data_wrapper<float, 2>;
template class point_cloud_data_wrapper<float, 3>;
template class point_cloud_data_wrapper<double, 2>;
template class point_cloud_data_wrapper<double, 3>;

// Explicit template instantiations for point_cloud_wrapper
template class point_cloud_wrapper<float, 2>;
template class point_cloud_wrapper<float, 3>;
template class point_cloud_wrapper<double, 2>;
template class point_cloud_wrapper<double, 3>;

} // namespace tf::py
