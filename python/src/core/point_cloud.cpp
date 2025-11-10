/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/core/point_cloud.hpp"

namespace tf::py {

auto register_point_cloud(nanobind::module_ &m) -> void {
  // Register PointCloud for float, 2D
  nanobind::class_<point_cloud_wrapper<float, 2>>(m, "PointCloudWrapperFloat2D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &point_cloud_wrapper<float, 2>::rebuild_tree)
      .def("ensure_tree", &point_cloud_wrapper<float, 2>::ensure_tree)
      .def("clear_tree", &point_cloud_wrapper<float, 2>::clear_tree)
      .def("has_tree", &point_cloud_wrapper<float, 2>::has_tree)
      .def("size", &point_cloud_wrapper<float, 2>::size)
      .def("dims", &point_cloud_wrapper<float, 2>::dims)
      .def("points_array", &point_cloud_wrapper<float, 2>::points_array)
      .def("has_transformation",
           &point_cloud_wrapper<float, 2>::has_transformation)
      .def("transformation", &point_cloud_wrapper<float, 2>::transformation)
      .def("set_transformation",
           &point_cloud_wrapper<float, 2>::set_transformation)
      .def("clear_transformation",
           &point_cloud_wrapper<float, 2>::clear_transformation);

  // Register PointCloud for float, 3D
  nanobind::class_<point_cloud_wrapper<float, 3>>(m, "PointCloudWrapperFloat3D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &point_cloud_wrapper<float, 3>::rebuild_tree)
      .def("ensure_tree", &point_cloud_wrapper<float, 3>::ensure_tree)
      .def("clear_tree", &point_cloud_wrapper<float, 3>::clear_tree)
      .def("has_tree", &point_cloud_wrapper<float, 3>::has_tree)
      .def("size", &point_cloud_wrapper<float, 3>::size)
      .def("dims", &point_cloud_wrapper<float, 3>::dims)
      .def("points_array", &point_cloud_wrapper<float, 3>::points_array)
      .def("has_transformation",
           &point_cloud_wrapper<float, 3>::has_transformation)
      .def("transformation", &point_cloud_wrapper<float, 3>::transformation)
      .def("set_transformation",
           &point_cloud_wrapper<float, 3>::set_transformation)
      .def("clear_transformation",
           &point_cloud_wrapper<float, 3>::clear_transformation);

  // Register PointCloud for double, 2D
  nanobind::class_<point_cloud_wrapper<double, 2>>(m,
                                                   "PointCloudWrapperDouble2D")
      .def(nanobind::init<nanobind::ndarray<nanobind::numpy, double,
                                            nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &point_cloud_wrapper<double, 2>::rebuild_tree)
      .def("ensure_tree", &point_cloud_wrapper<double, 2>::ensure_tree)
      .def("clear_tree", &point_cloud_wrapper<double, 2>::clear_tree)
      .def("has_tree", &point_cloud_wrapper<double, 2>::has_tree)
      .def("size", &point_cloud_wrapper<double, 2>::size)
      .def("dims", &point_cloud_wrapper<double, 2>::dims)
      .def("points_array", &point_cloud_wrapper<double, 2>::points_array)
      .def("has_transformation",
           &point_cloud_wrapper<double, 2>::has_transformation)
      .def("transformation", &point_cloud_wrapper<double, 2>::transformation)
      .def("set_transformation",
           &point_cloud_wrapper<double, 2>::set_transformation)
      .def("clear_transformation",
           &point_cloud_wrapper<double, 2>::clear_transformation);

  // Register PointCloud for double, 3D
  nanobind::class_<point_cloud_wrapper<double, 3>>(m,
                                                   "PointCloudWrapperDouble3D")
      .def(nanobind::init<nanobind::ndarray<nanobind::numpy, double,
                                            nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &point_cloud_wrapper<double, 3>::rebuild_tree)
      .def("ensure_tree", &point_cloud_wrapper<double, 3>::ensure_tree)
      .def("clear_tree", &point_cloud_wrapper<double, 3>::clear_tree)
      .def("has_tree", &point_cloud_wrapper<double, 3>::has_tree)
      .def("size", &point_cloud_wrapper<double, 3>::size)
      .def("dims", &point_cloud_wrapper<double, 3>::dims)
      .def("points_array", &point_cloud_wrapper<double, 3>::points_array)
      .def("has_transformation",
           &point_cloud_wrapper<double, 3>::has_transformation)
      .def("transformation", &point_cloud_wrapper<double, 3>::transformation)
      .def("set_transformation",
           &point_cloud_wrapper<double, 3>::set_transformation)
      .def("clear_transformation",
           &point_cloud_wrapper<double, 3>::clear_transformation);
}

// Explicit template instantiations
template class point_cloud_wrapper<float, 2>;
template class point_cloud_wrapper<float, 3>;
template class point_cloud_wrapper<double, 2>;
template class point_cloud_wrapper<double, 3>;

} // namespace tf::py
