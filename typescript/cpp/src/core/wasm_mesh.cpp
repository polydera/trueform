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

EMSCRIPTEN_BINDINGS(trueform_mesh) {
  emscripten::class_<tf::ts::wasm_mesh>("NativeMesh")
      .class_function("create", &tf::ts::wasm_mesh::create)
      .function("faces", &tf::ts::wasm_mesh::faces)
      .function("points", &tf::ts::wasm_mesh::points)
      .function("number_of_faces", &tf::ts::wasm_mesh::number_of_faces)
      .function("number_of_points", &tf::ts::wasm_mesh::number_of_points)
      .function("set_faces", &tf::ts::wasm_mesh::set_faces)
      .function("set_points", &tf::ts::wasm_mesh::set_points)
      .function("shallow_copy", &tf::ts::wasm_mesh::shallow_copy)
      .function("face_membership", &tf::ts::wasm_mesh::face_membership)
      .function("manifold_edge_link", &tf::ts::wasm_mesh::manifold_edge_link)
      .function("face_link", &tf::ts::wasm_mesh::face_link)
      .function("vertex_link", &tf::ts::wasm_mesh::vertex_link)
      .function("normals", &tf::ts::wasm_mesh::normals)
      .function("point_normals", &tf::ts::wasm_mesh::point_normals)
      .function("set_face_membership", &tf::ts::wasm_mesh::set_face_membership)
      .function("set_vertex_link", &tf::ts::wasm_mesh::set_vertex_link)
      .function("set_face_link", &tf::ts::wasm_mesh::set_face_link)
      .function("set_manifold_edge_link", &tf::ts::wasm_mesh::set_manifold_edge_link)
      .function("set_normals", &tf::ts::wasm_mesh::set_normals)
      .function("set_point_normals", &tf::ts::wasm_mesh::set_point_normals)
      .function("has_transformation", &tf::ts::wasm_mesh::has_transformation)
      .function("transformation", &tf::ts::wasm_mesh::transformation)
      .function("set_transformation", &tf::ts::wasm_mesh::set_transformation)
      .function("clear_transformation", &tf::ts::wasm_mesh::clear_transformation)
      .function("build_tree", &tf::ts::wasm_mesh::build_tree)
      .function("destroy", &tf::ts::wasm_mesh::destroy)
      .function("is_valid", &tf::ts::wasm_mesh::is_valid)
      // -- Cache state inspectors (diagnostic) --
      .function("is_tree_built", &tf::ts::wasm_mesh::is_tree_built)
      .function("is_tree_fresh", &tf::ts::wasm_mesh::is_tree_fresh)
      .function("is_half_edges_built", &tf::ts::wasm_mesh::is_half_edges_built)
      .function("is_half_edges_fresh", &tf::ts::wasm_mesh::is_half_edges_fresh)
      .function("is_face_membership_built", &tf::ts::wasm_mesh::is_face_membership_built)
      .function("is_face_membership_fresh", &tf::ts::wasm_mesh::is_face_membership_fresh)
      .function("is_manifold_edge_link_built", &tf::ts::wasm_mesh::is_manifold_edge_link_built)
      .function("is_manifold_edge_link_fresh", &tf::ts::wasm_mesh::is_manifold_edge_link_fresh)
      .function("is_face_link_built", &tf::ts::wasm_mesh::is_face_link_built)
      .function("is_face_link_fresh", &tf::ts::wasm_mesh::is_face_link_fresh)
      .function("is_vertex_link_built", &tf::ts::wasm_mesh::is_vertex_link_built)
      .function("is_vertex_link_fresh", &tf::ts::wasm_mesh::is_vertex_link_fresh)
      .function("is_normals_built", &tf::ts::wasm_mesh::is_normals_built)
      .function("is_normals_fresh", &tf::ts::wasm_mesh::is_normals_fresh)
      .function("is_point_normals_built", &tf::ts::wasm_mesh::is_point_normals_built)
      .function("is_point_normals_fresh", &tf::ts::wasm_mesh::is_point_normals_fresh);

  emscripten::register_vector<tf::ts::wasm_mesh>("VectorMesh");
}
