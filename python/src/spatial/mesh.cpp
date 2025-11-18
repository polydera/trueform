/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/spatial/mesh.hpp"

namespace tf::py {

auto register_mesh(nanobind::module_ &m) -> void {
  // int32, float, tri, 2D
  nanobind::class_<mesh_wrapper<int, float, 3, 2>>(m, "MeshWrapperIntFloat32D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int, float, 3, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, float, 3, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, float, 3, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, float, 3, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, float, 3, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, float, 3, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, float, 3, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, float, 3, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, float, 3, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, float, 3, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, float, 3, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, float, 3, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, float, 3, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, float, 3, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, float, 3, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, float, 3, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, float, 3, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, float, 3, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, float, 3, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, float, 3, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, float, 3, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, float, 3, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, float, 3, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, float, 3, 2>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, float, 3, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int, float, 3, 2>::dims)
      .def("faces_array", &mesh_wrapper<int, float, 3, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int, float, 3, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, float, 3, 2>::has_transformation)
      .def("transformation", &mesh_wrapper<int, float, 3, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, float, 3, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, float, 3, 2>::clear_transformation);

  // int32, float, tri, 3D
  nanobind::class_<mesh_wrapper<int, float, 3, 3>>(m, "MeshWrapperIntFloat33D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int, float, 3, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, float, 3, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, float, 3, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, float, 3, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, float, 3, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, float, 3, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, float, 3, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, float, 3, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, float, 3, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, float, 3, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, float, 3, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, float, 3, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, float, 3, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, float, 3, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, float, 3, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, float, 3, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, float, 3, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, float, 3, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, float, 3, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, float, 3, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, float, 3, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, float, 3, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, float, 3, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, float, 3, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, float, 3, 3>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, float, 3, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int, float, 3, 3>::dims)
      .def("faces_array", &mesh_wrapper<int, float, 3, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int, float, 3, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, float, 3, 3>::has_transformation)
      .def("transformation", &mesh_wrapper<int, float, 3, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, float, 3, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, float, 3, 3>::clear_transformation);

  // int32, float, quad, 2D
  nanobind::class_<mesh_wrapper<int, float, 4, 2>>(m, "MeshWrapperIntFloat42D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int, float, 4, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, float, 4, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, float, 4, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, float, 4, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, float, 4, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, float, 4, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, float, 4, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, float, 4, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, float, 4, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, float, 4, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, float, 4, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, float, 4, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, float, 4, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, float, 4, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, float, 4, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, float, 4, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, float, 4, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, float, 4, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, float, 4, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, float, 4, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, float, 4, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, float, 4, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, float, 4, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, float, 4, 2>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, float, 4, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int, float, 4, 2>::dims)
      .def("faces_array", &mesh_wrapper<int, float, 4, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int, float, 4, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, float, 4, 2>::has_transformation)
      .def("transformation", &mesh_wrapper<int, float, 4, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, float, 4, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, float, 4, 2>::clear_transformation);

  // int32, float, quad, 3D
  nanobind::class_<mesh_wrapper<int, float, 4, 3>>(m, "MeshWrapperIntFloat43D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int, float, 4, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, float, 4, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, float, 4, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, float, 4, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, float, 4, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, float, 4, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, float, 4, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, float, 4, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, float, 4, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, float, 4, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, float, 4, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, float, 4, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, float, 4, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, float, 4, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, float, 4, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, float, 4, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, float, 4, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, float, 4, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, float, 4, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, float, 4, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, float, 4, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, float, 4, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, float, 4, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, float, 4, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, float, 4, 3>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, float, 4, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int, float, 4, 3>::dims)
      .def("faces_array", &mesh_wrapper<int, float, 4, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int, float, 4, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, float, 4, 3>::has_transformation)
      .def("transformation", &mesh_wrapper<int, float, 4, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, float, 4, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, float, 4, 3>::clear_transformation);

  // int32, double, tri, 2D
  nanobind::class_<mesh_wrapper<int, double, 3, 2>>(m, "MeshWrapperIntDouble32D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int, double, 3, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, double, 3, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, double, 3, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, double, 3, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, double, 3, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, double, 3, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, double, 3, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, double, 3, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, double, 3, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, double, 3, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, double, 3, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, double, 3, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, double, 3, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, double, 3, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, double, 3, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, double, 3, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, double, 3, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, double, 3, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, double, 3, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, double, 3, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, double, 3, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, double, 3, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, double, 3, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, double, 3, 2>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, double, 3, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int, double, 3, 2>::dims)
      .def("faces_array", &mesh_wrapper<int, double, 3, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int, double, 3, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, double, 3, 2>::has_transformation)
      .def("transformation", &mesh_wrapper<int, double, 3, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, double, 3, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, double, 3, 2>::clear_transformation);

  // int32, double, tri, 3D
  nanobind::class_<mesh_wrapper<int, double, 3, 3>>(m, "MeshWrapperIntDouble33D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int, double, 3, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, double, 3, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, double, 3, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, double, 3, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, double, 3, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, double, 3, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, double, 3, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, double, 3, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, double, 3, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, double, 3, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, double, 3, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, double, 3, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, double, 3, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, double, 3, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, double, 3, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, double, 3, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, double, 3, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, double, 3, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, double, 3, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, double, 3, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, double, 3, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, double, 3, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, double, 3, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, double, 3, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, double, 3, 3>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, double, 3, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int, double, 3, 3>::dims)
      .def("faces_array", &mesh_wrapper<int, double, 3, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int, double, 3, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, double, 3, 3>::has_transformation)
      .def("transformation", &mesh_wrapper<int, double, 3, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, double, 3, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, double, 3, 3>::clear_transformation);

  // int32, double, quad, 2D
  nanobind::class_<mesh_wrapper<int, double, 4, 2>>(m, "MeshWrapperIntDouble42D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int, double, 4, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, double, 4, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, double, 4, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, double, 4, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, double, 4, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, double, 4, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, double, 4, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, double, 4, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, double, 4, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, double, 4, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, double, 4, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, double, 4, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, double, 4, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, double, 4, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, double, 4, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, double, 4, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, double, 4, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, double, 4, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, double, 4, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, double, 4, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, double, 4, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, double, 4, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, double, 4, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, double, 4, 2>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, double, 4, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int, double, 4, 2>::dims)
      .def("faces_array", &mesh_wrapper<int, double, 4, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int, double, 4, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, double, 4, 2>::has_transformation)
      .def("transformation", &mesh_wrapper<int, double, 4, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, double, 4, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, double, 4, 2>::clear_transformation);

  // int32, double, quad, 3D
  nanobind::class_<mesh_wrapper<int, double, 4, 3>>(m, "MeshWrapperIntDouble43D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int, double, 4, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int, double, 4, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int, double, 4, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int, double, 4, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int, double, 4, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int, double, 4, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int, double, 4, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int, double, 4, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int, double, 4, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int, double, 4, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int, double, 4, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int, double, 4, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int, double, 4, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int, double, 4, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int, double, 4, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int, double, 4, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int, double, 4, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int, double, 4, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int, double, 4, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int, double, 4, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int, double, 4, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int, double, 4, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int, double, 4, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int, double, 4, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int, double, 4, 3>::number_of_points)
      .def("number_of_faces", &mesh_wrapper<int, double, 4, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int, double, 4, 3>::dims)
      .def("faces_array", &mesh_wrapper<int, double, 4, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int, double, 4, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int, double, 4, 3>::has_transformation)
      .def("transformation", &mesh_wrapper<int, double, 4, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int, double, 4, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int, double, 4, 3>::clear_transformation);

  // int64, float, tri, 2D
  nanobind::class_<mesh_wrapper<int64_t, float, 3, 2>>(m, "MeshWrapperInt64Float32D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, float, 3, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, float, 3, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, float, 3, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, float, 3, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, float, 3, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, float, 3, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, float, 3, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, float, 3, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, float, 3, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, float, 3, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, float, 3, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, float, 3, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, float, 3, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, float, 3, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, float, 3, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, float, 3, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, float, 3, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, float, 3, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, float, 3, 2>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, float, 3, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, float, 3, 2>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, float, 3, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, float, 3, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, float, 3, 2>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, float, 3, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, float, 3, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, float, 3, 2>::clear_transformation);

  // int64, float, tri, 3D
  nanobind::class_<mesh_wrapper<int64_t, float, 3, 3>>(m, "MeshWrapperInt64Float33D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, float, 3, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, float, 3, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, float, 3, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, float, 3, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, float, 3, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, float, 3, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, float, 3, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, float, 3, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, float, 3, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, float, 3, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, float, 3, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 3, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, float, 3, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, float, 3, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, float, 3, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, float, 3, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, float, 3, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, float, 3, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, float, 3, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, float, 3, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, float, 3, 3>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, float, 3, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, float, 3, 3>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, float, 3, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, float, 3, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, float, 3, 3>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, float, 3, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, float, 3, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, float, 3, 3>::clear_transformation);

  // int64, float, quad, 2D
  nanobind::class_<mesh_wrapper<int64_t, float, 4, 2>>(m, "MeshWrapperInt64Float42D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, float, 4, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, float, 4, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, float, 4, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, float, 4, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, float, 4, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, float, 4, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, float, 4, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, float, 4, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, float, 4, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, float, 4, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, float, 4, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, float, 4, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, float, 4, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, float, 4, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, float, 4, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, float, 4, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, float, 4, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, float, 4, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, float, 4, 2>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, float, 4, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, float, 4, 2>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, float, 4, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, float, 4, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, float, 4, 2>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, float, 4, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, float, 4, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, float, 4, 2>::clear_transformation);

  // int64, float, quad, 3D
  nanobind::class_<mesh_wrapper<int64_t, float, 4, 3>>(m, "MeshWrapperInt64Float43D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, float, 4, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, float, 4, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, float, 4, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, float, 4, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, float, 4, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, float, 4, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, float, 4, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, float, 4, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, float, 4, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, float, 4, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, float, 4, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, float, 4, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, float, 4, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, float, 4, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, float, 4, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, float, 4, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, float, 4, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, float, 4, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, float, 4, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, float, 4, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, float, 4, 3>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, float, 4, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, float, 4, 3>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, float, 4, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, float, 4, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, float, 4, 3>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, float, 4, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, float, 4, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, float, 4, 3>::clear_transformation);

  // int64, double, tri, 2D
  nanobind::class_<mesh_wrapper<int64_t, double, 3, 2>>(m, "MeshWrapperInt64Double32D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, double, 3, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, double, 3, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, double, 3, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, double, 3, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, double, 3, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, double, 3, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, double, 3, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, double, 3, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, double, 3, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, double, 3, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, double, 3, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, double, 3, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, double, 3, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, double, 3, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, double, 3, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, double, 3, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, double, 3, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, double, 3, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, double, 3, 2>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, double, 3, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, double, 3, 2>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, double, 3, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, double, 3, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, double, 3, 2>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, double, 3, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, double, 3, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, double, 3, 2>::clear_transformation);

  // int64, double, tri, 3D
  nanobind::class_<mesh_wrapper<int64_t, double, 3, 3>>(m, "MeshWrapperInt64Double33D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 3>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, double, 3, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, double, 3, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, double, 3, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, double, 3, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, double, 3, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, double, 3, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, double, 3, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, double, 3, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, double, 3, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, double, 3, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, double, 3, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 3, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, double, 3, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, double, 3, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, double, 3, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, double, 3, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, double, 3, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, double, 3, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, double, 3, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, double, 3, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, double, 3, 3>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, double, 3, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, double, 3, 3>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, double, 3, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, double, 3, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, double, 3, 3>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, double, 3, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, double, 3, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, double, 3, 3>::clear_transformation);

  // int64, double, quad, 2D
  nanobind::class_<mesh_wrapper<int64_t, double, 4, 2>>(m, "MeshWrapperInt64Double42D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 2>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, double, 4, 2>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, double, 4, 2>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, double, 4, 2>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, double, 4, 2>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, double, 4, 2>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, double, 4, 2>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, double, 4, 2>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, double, 4, 2>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, double, 4, 2>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, double, 4, 2>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 2>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 2>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 2>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 2>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, double, 4, 2>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 2>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, double, 4, 2>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, double, 4, 2>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, double, 4, 2>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, double, 4, 2>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, double, 4, 2>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, double, 4, 2>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 2>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 2>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 2>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 2>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, double, 4, 2>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 2>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, double, 4, 2>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, double, 4, 2>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, double, 4, 2>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, double, 4, 2>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, double, 4, 2>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, double, 4, 2>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, double, 4, 2>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, double, 4, 2>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, double, 4, 2>::clear_transformation);

  // int64, double, quad, 3D
  nanobind::class_<mesh_wrapper<int64_t, double, 4, 3>>(m, "MeshWrapperInt64Double43D")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1, 4>>,
           nanobind::ndarray<nanobind::numpy, double,
                             nanobind::shape<-1, 3>>>())
      .def("rebuild_tree", &mesh_wrapper<int64_t, double, 4, 3>::rebuild_tree)
      .def("ensure_tree", &mesh_wrapper<int64_t, double, 4, 3>::ensure_tree)
      .def("clear_tree", &mesh_wrapper<int64_t, double, 4, 3>::clear_tree)
      .def("has_tree", &mesh_wrapper<int64_t, double, 4, 3>::has_tree)
      .def("rebuild_face_membership",
           &mesh_wrapper<int64_t, double, 4, 3>::rebuild_face_membership)
      .def("ensure_face_membership",
           &mesh_wrapper<int64_t, double, 4, 3>::ensure_face_membership)
      .def("clear_face_membership",
           &mesh_wrapper<int64_t, double, 4, 3>::clear_face_membership)
      .def("has_face_membership",
           &mesh_wrapper<int64_t, double, 4, 3>::has_face_membership)
      .def("face_membership_array",
           &mesh_wrapper<int64_t, double, 4, 3>::face_membership_array)
      .def("set_face_membership",
           &mesh_wrapper<int64_t, double, 4, 3>::set_face_membership)
      .def("rebuild_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 3>::rebuild_manifold_edge_link)
      .def("ensure_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 3>::ensure_manifold_edge_link)
      .def("clear_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 3>::clear_manifold_edge_link)
      .def("has_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 3>::has_manifold_edge_link)
      .def("manifold_edge_link_array",
           &mesh_wrapper<int64_t, double, 4, 3>::manifold_edge_link_array)
      .def("set_manifold_edge_link",
           &mesh_wrapper<int64_t, double, 4, 3>::set_manifold_edge_link)
      .def("rebuild_face_link",
           &mesh_wrapper<int64_t, double, 4, 3>::rebuild_face_link)
      .def("ensure_face_link",
           &mesh_wrapper<int64_t, double, 4, 3>::ensure_face_link)
      .def("clear_face_link",
           &mesh_wrapper<int64_t, double, 4, 3>::clear_face_link)
      .def("has_face_link",
           &mesh_wrapper<int64_t, double, 4, 3>::has_face_link)
      .def("face_link_array",
           &mesh_wrapper<int64_t, double, 4, 3>::face_link_array)
      .def("set_face_link",
           &mesh_wrapper<int64_t, double, 4, 3>::set_face_link)
      .def("rebuild_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 3>::rebuild_vertex_link)
      .def("ensure_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 3>::ensure_vertex_link)
      .def("clear_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 3>::clear_vertex_link)
      .def("has_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 3>::has_vertex_link)
      .def("vertex_link_array",
           &mesh_wrapper<int64_t, double, 4, 3>::vertex_link_array)
      .def("set_vertex_link",
           &mesh_wrapper<int64_t, double, 4, 3>::set_vertex_link)
      .def("number_of_points",
           &mesh_wrapper<int64_t, double, 4, 3>::number_of_points)
      .def("number_of_faces",
           &mesh_wrapper<int64_t, double, 4, 3>::number_of_faces)
      .def("dims", &mesh_wrapper<int64_t, double, 4, 3>::dims)
      .def("faces_array", &mesh_wrapper<int64_t, double, 4, 3>::faces_array)
      .def("points_array", &mesh_wrapper<int64_t, double, 4, 3>::points_array)
      .def("has_transformation",
           &mesh_wrapper<int64_t, double, 4, 3>::has_transformation)
      .def("transformation",
           &mesh_wrapper<int64_t, double, 4, 3>::transformation)
      .def("set_transformation",
           &mesh_wrapper<int64_t, double, 4, 3>::set_transformation)
      .def("clear_transformation",
           &mesh_wrapper<int64_t, double, 4, 3>::clear_transformation);
}

// Explicit template instantiations
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

} // namespace tf::py
