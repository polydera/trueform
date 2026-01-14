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

#include "trueform/python/spatial/edge_mesh.hpp"

namespace tf::py {

// Macro to define bindings for edge_mesh_wrapper
#define REGISTER_EDGE_MESH_WRAPPER(Index, RealT, Dims, Name)                   \
  nanobind::class_<edge_mesh_wrapper<Index, RealT, Dims>>(m, Name)             \
      .def(nanobind::init<                                                     \
           nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>,  \
           nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>()) \
      .def("build_tree", &edge_mesh_wrapper<Index, RealT, Dims>::build_tree)   \
      .def("has_tree", &edge_mesh_wrapper<Index, RealT, Dims>::has_tree)       \
      .def("build_edge_membership",                                            \
           &edge_mesh_wrapper<Index, RealT, Dims>::build_edge_membership)      \
      .def("has_edge_membership",                                              \
           &edge_mesh_wrapper<Index, RealT, Dims>::has_edge_membership)        \
      .def("edge_membership_array",                                            \
           &edge_mesh_wrapper<Index, RealT, Dims>::edge_membership_array)      \
      .def("set_edge_membership",                                              \
           &edge_mesh_wrapper<Index, RealT, Dims>::set_edge_membership)        \
      .def("build_vertex_link",                                                \
           &edge_mesh_wrapper<Index, RealT, Dims>::build_vertex_link)          \
      .def("has_vertex_link",                                                  \
           &edge_mesh_wrapper<Index, RealT, Dims>::has_vertex_link)            \
      .def("vertex_link_array",                                                \
           &edge_mesh_wrapper<Index, RealT, Dims>::vertex_link_array)          \
      .def("set_vertex_link",                                                  \
           &edge_mesh_wrapper<Index, RealT, Dims>::set_vertex_link)            \
      .def("number_of_points",                                                 \
           &edge_mesh_wrapper<Index, RealT, Dims>::number_of_points)           \
      .def("number_of_edges",                                                  \
           &edge_mesh_wrapper<Index, RealT, Dims>::number_of_edges)            \
      .def("dims", &edge_mesh_wrapper<Index, RealT, Dims>::dims)               \
      .def("edges_array",                                                      \
           &edge_mesh_wrapper<Index, RealT, Dims>::edges_array)                \
      .def("set_edges_array",                                                  \
           &edge_mesh_wrapper<Index, RealT, Dims>::set_edges_array)            \
      .def("points_array",                                                     \
           &edge_mesh_wrapper<Index, RealT, Dims>::points_array)               \
      .def("set_points_array",                                                 \
           &edge_mesh_wrapper<Index, RealT, Dims>::set_points_array)           \
      .def("has_transformation",                                               \
           &edge_mesh_wrapper<Index, RealT, Dims>::has_transformation)         \
      .def("transformation",                                                   \
           &edge_mesh_wrapper<Index, RealT, Dims>::transformation)             \
      .def("set_transformation",                                               \
           &edge_mesh_wrapper<Index, RealT, Dims>::set_transformation)         \
      .def("clear_transformation",                                             \
           &edge_mesh_wrapper<Index, RealT, Dims>::clear_transformation)       \
      .def("mark_modified",                                                    \
           &edge_mesh_wrapper<Index, RealT, Dims>::mark_modified)              \
      .def("shared_view", &edge_mesh_wrapper<Index, RealT, Dims>::shared_view)

auto register_edge_mesh(nanobind::module_ &m) -> void {
  // int32
  REGISTER_EDGE_MESH_WRAPPER(int, float, 2, "EdgeMeshWrapperIntFloat2D");
  REGISTER_EDGE_MESH_WRAPPER(int, float, 3, "EdgeMeshWrapperIntFloat3D");
  REGISTER_EDGE_MESH_WRAPPER(int, double, 2, "EdgeMeshWrapperIntDouble2D");
  REGISTER_EDGE_MESH_WRAPPER(int, double, 3, "EdgeMeshWrapperIntDouble3D");

  // int64
  REGISTER_EDGE_MESH_WRAPPER(int64_t, float, 2, "EdgeMeshWrapperInt64Float2D");
  REGISTER_EDGE_MESH_WRAPPER(int64_t, float, 3, "EdgeMeshWrapperInt64Float3D");
  REGISTER_EDGE_MESH_WRAPPER(int64_t, double, 2, "EdgeMeshWrapperInt64Double2D");
  REGISTER_EDGE_MESH_WRAPPER(int64_t, double, 3, "EdgeMeshWrapperInt64Double3D");
}

#undef REGISTER_EDGE_MESH_WRAPPER

// Explicit template instantiations for edge_mesh_data_wrapper
template class edge_mesh_data_wrapper<int, float, 2>;
template class edge_mesh_data_wrapper<int, float, 3>;
template class edge_mesh_data_wrapper<int, double, 2>;
template class edge_mesh_data_wrapper<int, double, 3>;
template class edge_mesh_data_wrapper<int64_t, float, 2>;
template class edge_mesh_data_wrapper<int64_t, float, 3>;
template class edge_mesh_data_wrapper<int64_t, double, 2>;
template class edge_mesh_data_wrapper<int64_t, double, 3>;

// Explicit template instantiations for edge_mesh_wrapper
template class edge_mesh_wrapper<int, float, 2>;
template class edge_mesh_wrapper<int, float, 3>;
template class edge_mesh_wrapper<int, double, 2>;
template class edge_mesh_wrapper<int, double, 3>;
template class edge_mesh_wrapper<int64_t, float, 2>;
template class edge_mesh_wrapper<int64_t, float, 3>;
template class edge_mesh_wrapper<int64_t, double, 2>;
template class edge_mesh_wrapper<int64_t, double, 3>;

} // namespace tf::py
