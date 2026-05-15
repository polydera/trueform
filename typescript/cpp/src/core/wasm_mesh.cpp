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

#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>
#include <string>

namespace {

template <typename Real>
auto register_wasm_mesh_bindings(const std::string &class_name) -> void {
  using mesh_t = tf::ts::wasm_mesh<Real>;
  emscripten::class_<mesh_t>(class_name.c_str())
      .template constructor<>()
      .class_function("create", &mesh_t::create)
      .function("faces", &mesh_t::faces)
      .function("points", &mesh_t::points)
      .function("number_of_faces", &mesh_t::number_of_faces)
      .function("number_of_points", &mesh_t::number_of_points)
      .function("set_faces", &mesh_t::set_faces)
      .function("set_points", &mesh_t::set_points)
      .function("shallow_copy", &mesh_t::shallow_copy)
      .function("face_membership", &mesh_t::face_membership)
      .function("manifold_edge_link", &mesh_t::manifold_edge_link)
      .function("face_link", &mesh_t::face_link)
      .function("vertex_link", &mesh_t::vertex_link)
      .function("normals", &mesh_t::normals)
      .function("point_normals", &mesh_t::point_normals)
      .function("set_face_membership", &mesh_t::set_face_membership)
      .function("set_vertex_link", &mesh_t::set_vertex_link)
      .function("set_face_link", &mesh_t::set_face_link)
      .function("set_manifold_edge_link", &mesh_t::set_manifold_edge_link)
      .function("set_normals", &mesh_t::set_normals)
      .function("set_point_normals", &mesh_t::set_point_normals)
      .function("has_transformation", &mesh_t::has_transformation)
      .function("transformation", &mesh_t::transformation)
      .function("set_transformation", &mesh_t::set_transformation)
      .function("clear_transformation", &mesh_t::clear_transformation)
      .function("build_tree", &mesh_t::build_tree)
      .function("destroy", &mesh_t::destroy)
      .function("is_valid", &mesh_t::is_valid)
      .function("is_tree_built", &mesh_t::is_tree_built)
      .function("is_tree_fresh", &mesh_t::is_tree_fresh)
      .function("is_half_edges_built", &mesh_t::is_half_edges_built)
      .function("is_half_edges_fresh", &mesh_t::is_half_edges_fresh)
      .function("is_face_membership_built", &mesh_t::is_face_membership_built)
      .function("is_face_membership_fresh", &mesh_t::is_face_membership_fresh)
      .function("is_manifold_edge_link_built", &mesh_t::is_manifold_edge_link_built)
      .function("is_manifold_edge_link_fresh", &mesh_t::is_manifold_edge_link_fresh)
      .function("is_face_link_built", &mesh_t::is_face_link_built)
      .function("is_face_link_fresh", &mesh_t::is_face_link_fresh)
      .function("is_vertex_link_built", &mesh_t::is_vertex_link_built)
      .function("is_vertex_link_fresh", &mesh_t::is_vertex_link_fresh)
      .function("is_normals_built", &mesh_t::is_normals_built)
      .function("is_normals_fresh", &mesh_t::is_normals_fresh)
      .function("is_point_normals_built", &mesh_t::is_point_normals_built)
      .function("is_point_normals_fresh", &mesh_t::is_point_normals_fresh);
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_mesh) {
  register_wasm_mesh_bindings<float>("NativeFloat32Mesh");
  register_wasm_mesh_bindings<double>("NativeFloat64Mesh");

  emscripten::register_vector<tf::ts::wasm_mesh<float>>("VectorMesh");
}
