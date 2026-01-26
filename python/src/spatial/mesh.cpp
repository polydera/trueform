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

#include "trueform/python/spatial/mesh.hpp"

namespace tf::py {

// Macro to define bindings for fixed-size mesh wrapper
#define REGISTER_MESH_WRAPPER(Index, RealT, Ngon, Dims, Name)                  \
  nanobind::class_<mesh_wrapper<Index, RealT, Ngon, Dims>>(m, Name)            \
      .def(nanobind::init<                                                     \
           nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>, \
           nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>()) \
      .def("build_tree", &mesh_wrapper<Index, RealT, Ngon, Dims>::build_tree)  \
      .def("has_tree", &mesh_wrapper<Index, RealT, Ngon, Dims>::has_tree)      \
      .def("build_face_membership",                                            \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::build_face_membership)     \
      .def("has_face_membership",                                              \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_face_membership)       \
      .def("face_membership_array",                                            \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::face_membership_array)     \
      .def("set_face_membership",                                              \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_face_membership)       \
      .def("build_manifold_edge_link",                                         \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::build_manifold_edge_link)  \
      .def("has_manifold_edge_link",                                           \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_manifold_edge_link)    \
      .def("manifold_edge_link_array",                                         \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::manifold_edge_link_array)  \
      .def("set_manifold_edge_link",                                           \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_manifold_edge_link)    \
      .def("build_face_link",                                                  \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::build_face_link)           \
      .def("has_face_link",                                                    \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_face_link)             \
      .def("face_link_array",                                                  \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::face_link_array)           \
      .def("set_face_link",                                                    \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_face_link)             \
      .def("build_vertex_link",                                                \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::build_vertex_link)         \
      .def("has_vertex_link",                                                  \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_vertex_link)           \
      .def("vertex_link_array",                                                \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::vertex_link_array)         \
      .def("set_vertex_link",                                                  \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_vertex_link)           \
      .def("number_of_points",                                                 \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::number_of_points)          \
      .def("number_of_faces",                                                  \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::number_of_faces)           \
      .def("dims", &mesh_wrapper<Index, RealT, Ngon, Dims>::dims)              \
      .def("faces_array",                                                      \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::faces_array)               \
      .def("set_faces_array",                                                  \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_faces_array)           \
      .def("points_array",                                                     \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::points_array)              \
      .def("set_points_array",                                                 \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_points_array)          \
      .def("has_transformation",                                               \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_transformation)        \
      .def("transformation",                                                   \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::transformation)            \
      .def("set_transformation",                                               \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_transformation)        \
      .def("clear_transformation",                                             \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::clear_transformation)      \
      .def("mark_modified",                                                    \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::mark_modified)             \
      .def("shared_view", &mesh_wrapper<Index, RealT, Ngon, Dims>::shared_view) \
      .def("build_normals",                                                    \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::build_normals)             \
      .def("has_normals",                                                      \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_normals)               \
      .def("normals_array",                                                    \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::normals_array)             \
      .def("set_normals",                                                      \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_normals)               \
      .def("build_point_normals",                                              \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::build_point_normals)       \
      .def("has_point_normals",                                                \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::has_point_normals)         \
      .def("point_normals_array",                                              \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::point_normals_array)       \
      .def("set_point_normals",                                                \
           &mesh_wrapper<Index, RealT, Ngon, Dims>::set_point_normals)

// Macro to define bindings for dynamic-size mesh wrapper
#define REGISTER_MESH_WRAPPER_DYNAMIC(Index, RealT, Dims, Name)                \
  nanobind::class_<mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>>(m, Name) \
      .def(nanobind::init<                                                     \
           tf::py::offset_blocked_array_wrapper<Index, Index>,                 \
           nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>()) \
      .def("build_tree",                                                       \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_tree)    \
      .def("has_tree",                                                         \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_tree)      \
      .def("build_face_membership",                                            \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_face_membership) \
      .def("has_face_membership",                                              \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_face_membership) \
      .def("face_membership_array",                                            \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::face_membership_array) \
      .def("set_face_membership",                                              \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_face_membership) \
      .def("build_manifold_edge_link",                                         \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_manifold_edge_link) \
      .def("has_manifold_edge_link",                                           \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_manifold_edge_link) \
      .def("manifold_edge_link_array",                                         \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::manifold_edge_link_array) \
      .def("set_manifold_edge_link",                                           \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_manifold_edge_link) \
      .def("build_face_link",                                                  \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_face_link) \
      .def("has_face_link",                                                    \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_face_link) \
      .def("face_link_array",                                                  \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::face_link_array) \
      .def("set_face_link",                                                    \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_face_link) \
      .def("build_vertex_link",                                                \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_vertex_link) \
      .def("has_vertex_link",                                                  \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_vertex_link) \
      .def("vertex_link_array",                                                \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::vertex_link_array) \
      .def("set_vertex_link",                                                  \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_vertex_link) \
      .def("number_of_points",                                                 \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::number_of_points) \
      .def("number_of_faces",                                                  \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::number_of_faces) \
      .def("dims",                                                             \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::dims)          \
      .def("faces_array",                                                      \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::faces_array)   \
      .def("set_faces_array",                                                  \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_faces_array) \
      .def("points_array",                                                     \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::points_array)  \
      .def("set_points_array",                                                 \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_points_array) \
      .def("has_transformation",                                               \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_transformation) \
      .def("transformation",                                                   \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::transformation) \
      .def("set_transformation",                                               \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_transformation) \
      .def("clear_transformation",                                             \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::clear_transformation) \
      .def("mark_modified",                                                    \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::mark_modified) \
      .def("shared_view",                                                      \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::shared_view)   \
      .def("build_normals",                                                    \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_normals) \
      .def("has_normals",                                                      \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_normals)   \
      .def("normals_array",                                                    \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::normals_array) \
      .def("set_normals",                                                      \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_normals)   \
      .def("build_point_normals",                                              \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::build_point_normals) \
      .def("has_point_normals",                                                \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::has_point_normals) \
      .def("point_normals_array",                                              \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::point_normals_array) \
      .def("set_point_normals",                                                \
           &mesh_wrapper<Index, RealT, tf::dynamic_size, Dims>::set_point_normals)

auto register_mesh(nanobind::module_ &m) -> void {
  // Fixed-size meshes: int32
  REGISTER_MESH_WRAPPER(int, float, 3, 2, "MeshWrapperIntFloat32D");
  REGISTER_MESH_WRAPPER(int, float, 3, 3, "MeshWrapperIntFloat33D");
  REGISTER_MESH_WRAPPER(int, float, 4, 2, "MeshWrapperIntFloat42D");
  REGISTER_MESH_WRAPPER(int, float, 4, 3, "MeshWrapperIntFloat43D");
  REGISTER_MESH_WRAPPER(int, double, 3, 2, "MeshWrapperIntDouble32D");
  REGISTER_MESH_WRAPPER(int, double, 3, 3, "MeshWrapperIntDouble33D");
  REGISTER_MESH_WRAPPER(int, double, 4, 2, "MeshWrapperIntDouble42D");
  REGISTER_MESH_WRAPPER(int, double, 4, 3, "MeshWrapperIntDouble43D");

  // Fixed-size meshes: int64
  REGISTER_MESH_WRAPPER(int64_t, float, 3, 2, "MeshWrapperInt64Float32D");
  REGISTER_MESH_WRAPPER(int64_t, float, 3, 3, "MeshWrapperInt64Float33D");
  REGISTER_MESH_WRAPPER(int64_t, float, 4, 2, "MeshWrapperInt64Float42D");
  REGISTER_MESH_WRAPPER(int64_t, float, 4, 3, "MeshWrapperInt64Float43D");
  REGISTER_MESH_WRAPPER(int64_t, double, 3, 2, "MeshWrapperInt64Double32D");
  REGISTER_MESH_WRAPPER(int64_t, double, 3, 3, "MeshWrapperInt64Double33D");
  REGISTER_MESH_WRAPPER(int64_t, double, 4, 2, "MeshWrapperInt64Double42D");
  REGISTER_MESH_WRAPPER(int64_t, double, 4, 3, "MeshWrapperInt64Double43D");

  // Dynamic-size meshes: int32
  REGISTER_MESH_WRAPPER_DYNAMIC(int, float, 2, "MeshWrapperIntFloatDynamic2D");
  REGISTER_MESH_WRAPPER_DYNAMIC(int, float, 3, "MeshWrapperIntFloatDynamic3D");
  REGISTER_MESH_WRAPPER_DYNAMIC(int, double, 2, "MeshWrapperIntDoubleDynamic2D");
  REGISTER_MESH_WRAPPER_DYNAMIC(int, double, 3, "MeshWrapperIntDoubleDynamic3D");

  // Dynamic-size meshes: int64
  REGISTER_MESH_WRAPPER_DYNAMIC(int64_t, float, 2, "MeshWrapperInt64FloatDynamic2D");
  REGISTER_MESH_WRAPPER_DYNAMIC(int64_t, float, 3, "MeshWrapperInt64FloatDynamic3D");
  REGISTER_MESH_WRAPPER_DYNAMIC(int64_t, double, 2, "MeshWrapperInt64DoubleDynamic2D");
  REGISTER_MESH_WRAPPER_DYNAMIC(int64_t, double, 3, "MeshWrapperInt64DoubleDynamic3D");
}

#undef REGISTER_MESH_WRAPPER
#undef REGISTER_MESH_WRAPPER_DYNAMIC

// Explicit template instantiations for fixed-size mesh_data_wrapper
template class mesh_data_wrapper<int, float, 3, 2>;
template class mesh_data_wrapper<int, float, 3, 3>;
template class mesh_data_wrapper<int, float, 4, 2>;
template class mesh_data_wrapper<int, float, 4, 3>;
template class mesh_data_wrapper<int, double, 3, 2>;
template class mesh_data_wrapper<int, double, 3, 3>;
template class mesh_data_wrapper<int, double, 4, 2>;
template class mesh_data_wrapper<int, double, 4, 3>;
template class mesh_data_wrapper<int64_t, float, 3, 2>;
template class mesh_data_wrapper<int64_t, float, 3, 3>;
template class mesh_data_wrapper<int64_t, float, 4, 2>;
template class mesh_data_wrapper<int64_t, float, 4, 3>;
template class mesh_data_wrapper<int64_t, double, 3, 2>;
template class mesh_data_wrapper<int64_t, double, 3, 3>;
template class mesh_data_wrapper<int64_t, double, 4, 2>;
template class mesh_data_wrapper<int64_t, double, 4, 3>;

// Explicit template instantiations for dynamic-size mesh_data_wrapper
template class mesh_data_wrapper<int, float, tf::dynamic_size, 2>;
template class mesh_data_wrapper<int, float, tf::dynamic_size, 3>;
template class mesh_data_wrapper<int, double, tf::dynamic_size, 2>;
template class mesh_data_wrapper<int, double, tf::dynamic_size, 3>;
template class mesh_data_wrapper<int64_t, float, tf::dynamic_size, 2>;
template class mesh_data_wrapper<int64_t, float, tf::dynamic_size, 3>;
template class mesh_data_wrapper<int64_t, double, tf::dynamic_size, 2>;
template class mesh_data_wrapper<int64_t, double, tf::dynamic_size, 3>;

// Explicit template instantiations for fixed-size mesh_wrapper
template class mesh_wrapper<int, float, 3, 2>;
template class mesh_wrapper<int, float, 3, 3>;
template class mesh_wrapper<int, float, 4, 2>;
template class mesh_wrapper<int, float, 4, 3>;
template class mesh_wrapper<int, double, 3, 2>;
template class mesh_wrapper<int, double, 3, 3>;
template class mesh_wrapper<int, double, 4, 2>;
template class mesh_wrapper<int, double, 4, 3>;
template class mesh_wrapper<int64_t, float, 3, 2>;
template class mesh_wrapper<int64_t, float, 3, 3>;
template class mesh_wrapper<int64_t, float, 4, 2>;
template class mesh_wrapper<int64_t, float, 4, 3>;
template class mesh_wrapper<int64_t, double, 3, 2>;
template class mesh_wrapper<int64_t, double, 3, 3>;
template class mesh_wrapper<int64_t, double, 4, 2>;
template class mesh_wrapper<int64_t, double, 4, 3>;

// Explicit template instantiations for dynamic-size mesh_wrapper
template class mesh_wrapper<int, float, tf::dynamic_size, 2>;
template class mesh_wrapper<int, float, tf::dynamic_size, 3>;
template class mesh_wrapper<int, double, tf::dynamic_size, 2>;
template class mesh_wrapper<int, double, tf::dynamic_size, 3>;
template class mesh_wrapper<int64_t, float, tf::dynamic_size, 2>;
template class mesh_wrapper<int64_t, float, tf::dynamic_size, 3>;
template class mesh_wrapper<int64_t, double, tf::dynamic_size, 2>;
template class mesh_wrapper<int64_t, double, tf::dynamic_size, 3>;

} // namespace tf::py
