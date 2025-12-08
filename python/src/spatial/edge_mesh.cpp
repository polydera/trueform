/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/spatial/edge_mesh.hpp"

namespace tf::py {

auto register_edge_mesh(nanobind::module_ &m) -> void {
  // int32, float, 2D
  nanobind::class_<edge_mesh_wrapper<int, float, 2>>(m,
                                                      "EdgeMeshWrapperIntFloat2D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &edge_mesh_wrapper<int, float, 2>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int, float, 2>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int, float, 2>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int, float, 2>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int, float, 2>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int, float, 2>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int, float, 2>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int, float, 2>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int, float, 2>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int, float, 2>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int, float, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int, float, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int, float, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int, float, 2>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int, float, 2>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int, float, 2>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int, float, 2>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int, float, 2>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int, float, 2>::dims)
      .def("edges_array", &edge_mesh_wrapper<int, float, 2>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int, float, 2>::set_edges_array)
      .def("points_array", &edge_mesh_wrapper<int, float, 2>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int, float, 2>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int, float, 2>::has_transformation)
      .def("transformation", &edge_mesh_wrapper<int, float, 2>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int, float, 2>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int, float, 2>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int, float, 2>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int, float, 2>::shared_view);

  // int32, float, 3D
  nanobind::class_<edge_mesh_wrapper<int, float, 3>>(m,
                                                      "EdgeMeshWrapperIntFloat3D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &edge_mesh_wrapper<int, float, 3>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int, float, 3>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int, float, 3>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int, float, 3>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int, float, 3>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int, float, 3>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int, float, 3>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int, float, 3>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int, float, 3>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int, float, 3>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int, float, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int, float, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int, float, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int, float, 3>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int, float, 3>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int, float, 3>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int, float, 3>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int, float, 3>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int, float, 3>::dims)
      .def("edges_array", &edge_mesh_wrapper<int, float, 3>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int, float, 3>::set_edges_array)
      .def("points_array", &edge_mesh_wrapper<int, float, 3>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int, float, 3>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int, float, 3>::has_transformation)
      .def("transformation", &edge_mesh_wrapper<int, float, 3>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int, float, 3>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int, float, 3>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int, float, 3>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int, float, 3>::shared_view);

  // int32, double, 2D
  nanobind::class_<edge_mesh_wrapper<int, double, 2>>(
      m, "EdgeMeshWrapperIntDouble2D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &edge_mesh_wrapper<int, double, 2>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int, double, 2>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int, double, 2>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int, double, 2>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int, double, 2>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int, double, 2>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int, double, 2>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int, double, 2>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int, double, 2>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int, double, 2>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int, double, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int, double, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int, double, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int, double, 2>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int, double, 2>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int, double, 2>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int, double, 2>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int, double, 2>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int, double, 2>::dims)
      .def("edges_array", &edge_mesh_wrapper<int, double, 2>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int, double, 2>::set_edges_array)
      .def("points_array", &edge_mesh_wrapper<int, double, 2>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int, double, 2>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int, double, 2>::has_transformation)
      .def("transformation",
           &edge_mesh_wrapper<int, double, 2>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int, double, 2>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int, double, 2>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int, double, 2>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int, double, 2>::shared_view);

  // int32, double, 3D
  nanobind::class_<edge_mesh_wrapper<int, double, 3>>(
      m, "EdgeMeshWrapperIntDouble3D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &edge_mesh_wrapper<int, double, 3>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int, double, 3>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int, double, 3>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int, double, 3>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int, double, 3>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int, double, 3>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int, double, 3>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int, double, 3>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int, double, 3>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int, double, 3>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int, double, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int, double, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int, double, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int, double, 3>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int, double, 3>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int, double, 3>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int, double, 3>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int, double, 3>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int, double, 3>::dims)
      .def("edges_array", &edge_mesh_wrapper<int, double, 3>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int, double, 3>::set_edges_array)
      .def("points_array", &edge_mesh_wrapper<int, double, 3>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int, double, 3>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int, double, 3>::has_transformation)
      .def("transformation",
           &edge_mesh_wrapper<int, double, 3>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int, double, 3>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int, double, 3>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int, double, 3>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int, double, 3>::shared_view);

  // int64, float, 2D
  nanobind::class_<edge_mesh_wrapper<int64_t, float, 2>>(
      m, "EdgeMeshWrapperInt64Float2D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &edge_mesh_wrapper<int64_t, float, 2>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int64_t, float, 2>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int64_t, float, 2>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int64_t, float, 2>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 2>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 2>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 2>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 2>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int64_t, float, 2>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 2>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 2>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int64_t, float, 2>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 2>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int64_t, float, 2>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int64_t, float, 2>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int64_t, float, 2>::dims)
      .def("edges_array", &edge_mesh_wrapper<int64_t, float, 2>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int64_t, float, 2>::set_edges_array)
      .def("points_array",
           &edge_mesh_wrapper<int64_t, float, 2>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int64_t, float, 2>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int64_t, float, 2>::has_transformation)
      .def("transformation",
           &edge_mesh_wrapper<int64_t, float, 2>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int64_t, float, 2>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int64_t, float, 2>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int64_t, float, 2>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int64_t, float, 2>::shared_view);

  // int64, float, 3D
  nanobind::class_<edge_mesh_wrapper<int64_t, float, 3>>(
      m, "EdgeMeshWrapperInt64Float3D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &edge_mesh_wrapper<int64_t, float, 3>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int64_t, float, 3>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int64_t, float, 3>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int64_t, float, 3>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 3>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 3>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 3>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 3>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int64_t, float, 3>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int64_t, float, 3>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 3>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int64_t, float, 3>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int64_t, float, 3>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int64_t, float, 3>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int64_t, float, 3>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int64_t, float, 3>::dims)
      .def("edges_array", &edge_mesh_wrapper<int64_t, float, 3>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int64_t, float, 3>::set_edges_array)
      .def("points_array",
           &edge_mesh_wrapper<int64_t, float, 3>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int64_t, float, 3>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int64_t, float, 3>::has_transformation)
      .def("transformation",
           &edge_mesh_wrapper<int64_t, float, 3>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int64_t, float, 3>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int64_t, float, 3>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int64_t, float, 3>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int64_t, float, 3>::shared_view);

  // int64, double, 2D
  nanobind::class_<edge_mesh_wrapper<int64_t, double, 2>>(
      m, "EdgeMeshWrapperInt64Double2D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 2>>>())
      .def("rebuild_tree",
           &edge_mesh_wrapper<int64_t, double, 2>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int64_t, double, 2>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int64_t, double, 2>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int64_t, double, 2>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 2>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 2>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 2>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 2>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int64_t, double, 2>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 2>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 2>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int64_t, double, 2>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 2>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int64_t, double, 2>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int64_t, double, 2>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int64_t, double, 2>::dims)
      .def("edges_array",
           &edge_mesh_wrapper<int64_t, double, 2>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int64_t, double, 2>::set_edges_array)
      .def("points_array",
           &edge_mesh_wrapper<int64_t, double, 2>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int64_t, double, 2>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int64_t, double, 2>::has_transformation)
      .def("transformation",
           &edge_mesh_wrapper<int64_t, double, 2>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int64_t, double, 2>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int64_t, double, 2>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int64_t, double, 2>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int64_t, double, 2>::shared_view);

  // int64, double, 3D
  nanobind::class_<edge_mesh_wrapper<int64_t, double, 3>>(
      m, "EdgeMeshWrapperInt64Double3D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 2>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 3>>>())
      .def("rebuild_tree",
           &edge_mesh_wrapper<int64_t, double, 3>::rebuild_tree)
      .def("ensure_tree", &edge_mesh_wrapper<int64_t, double, 3>::ensure_tree)
      .def("clear_tree", &edge_mesh_wrapper<int64_t, double, 3>::clear_tree)
      .def("has_tree", &edge_mesh_wrapper<int64_t, double, 3>::has_tree)
      .def("rebuild_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 3>::rebuild_edge_membership)
      .def("ensure_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 3>::ensure_edge_membership)
      .def("clear_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 3>::clear_edge_membership)
      .def("has_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 3>::has_edge_membership)
      .def("edge_membership_array",
           &edge_mesh_wrapper<int64_t, double, 3>::edge_membership_array)
      .def("set_edge_membership",
           &edge_mesh_wrapper<int64_t, double, 3>::set_edge_membership)
      .def("rebuild_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 3>::has_vertex_link)
      .def("vertex_link_array",
           &edge_mesh_wrapper<int64_t, double, 3>::vertex_link_array)
      .def("set_vertex_link",
           &edge_mesh_wrapper<int64_t, double, 3>::set_vertex_link)
      .def("number_of_points",
           &edge_mesh_wrapper<int64_t, double, 3>::number_of_points)
      .def("number_of_edges",
           &edge_mesh_wrapper<int64_t, double, 3>::number_of_edges)
      .def("dims", &edge_mesh_wrapper<int64_t, double, 3>::dims)
      .def("edges_array",
           &edge_mesh_wrapper<int64_t, double, 3>::edges_array)
      .def("set_edges_array", &edge_mesh_wrapper<int64_t, double, 3>::set_edges_array)
      .def("points_array",
           &edge_mesh_wrapper<int64_t, double, 3>::points_array)
      .def("set_points_array", &edge_mesh_wrapper<int64_t, double, 3>::set_points_array)
      .def("has_transformation",
           &edge_mesh_wrapper<int64_t, double, 3>::has_transformation)
      .def("transformation",
           &edge_mesh_wrapper<int64_t, double, 3>::transformation)
      .def("set_transformation",
           &edge_mesh_wrapper<int64_t, double, 3>::set_transformation)
      .def("clear_transformation",
           &edge_mesh_wrapper<int64_t, double, 3>::clear_transformation)
      .def("mark_modified", &edge_mesh_wrapper<int64_t, double, 3>::mark_modified)
      .def("shared_view", &edge_mesh_wrapper<int64_t, double, 3>::shared_view);
}

// Explicit template instantiations for edge_mesh_data_wrapper
template class edge_mesh_data_wrapper<int, float, 2>;
template class edge_mesh_data_wrapper<int, float, 3>;
template class edge_mesh_data_wrapper<int, double, 2>;
template class edge_mesh_data_wrapper<int, double, 3>;
template class edge_mesh_data_wrapper<int64_t, float, 2>;
template class edge_mesh_data_wrapper<int64_t, float, 3>;
template class edge_mesh_data_wrapper<int64_t, double, 2>;
template class edge_mesh_data_wrapper<int64_t, double, 3>;

// Explicit template instantiations
template class edge_mesh_wrapper<int, float, 2>;
template class edge_mesh_wrapper<int, float, 3>;
template class edge_mesh_wrapper<int, double, 2>;
template class edge_mesh_wrapper<int, double, 3>;
template class edge_mesh_wrapper<int64_t, float, 2>;
template class edge_mesh_wrapper<int64_t, float, 3>;
template class edge_mesh_wrapper<int64_t, double, 2>;
template class edge_mesh_wrapper<int64_t, double, 3>;

} // namespace tf::py
