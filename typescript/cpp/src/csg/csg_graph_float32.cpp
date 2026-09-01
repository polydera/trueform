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

#include "./csg_graph_impl.hpp"

EMSCRIPTEN_BINDINGS(trueform_csg_graph_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::class_<wasm_csg_graph<Real>>("CsgGraphFloat32")
      .constructor<>()
      .function("isValid", &wasm_csg_graph<Real>::is_valid)
      .function("destroy", &wasm_csg_graph<Real>::destroy);

  emscripten::value_object<csg_mesh_labeled_t<Real>>("CsgMeshLabeledFloat32")
      .field("mesh", &csg_mesh_labeled_t<Real>::mesh)
      .field("tagLabels", &csg_mesh_labeled_t<Real>::tag_labels)
      .field("faceLabels", &csg_mesh_labeled_t<Real>::face_labels);

  emscripten::value_object<csg_mesh_index_map_t<Real>>(
      "CsgMeshIndexMapFloat32")
      .field("mesh", &csg_mesh_index_map_t<Real>::mesh)
      .field("pointTagLabels", &csg_mesh_index_map_t<Real>::point_tag_labels)
      .field("pointLabels", &csg_mesh_index_map_t<Real>::point_labels)
      .field("faceTagLabels", &csg_mesh_index_map_t<Real>::face_tag_labels)
      .field("faceLabels", &csg_mesh_index_map_t<Real>::face_labels)
      .field("pointFOffsets", &csg_mesh_index_map_t<Real>::point_f_offsets)
      .field("pointFData", &csg_mesh_index_map_t<Real>::point_f_data)
      .field("nOriginalPoints",
             &csg_mesh_index_map_t<Real>::n_original_points)
      .field("uncutFaces", &csg_mesh_index_map_t<Real>::uncut_faces)
      .field("nTags", &csg_mesh_index_map_t<Real>::n_tags)
      .field("nOutputPoints", &csg_mesh_index_map_t<Real>::n_output_points);

  emscripten::value_object<csg_domains_t<Real>>("CsgDomainsFloat32")
      .field("meshes", &csg_domains_t<Real>::meshes)
      .field("ids", &csg_domains_t<Real>::ids);

  emscripten::value_object<csg_domains_labeled_t<Real>>(
      "CsgDomainsLabeledFloat32")
      .field("meshes", &csg_domains_labeled_t<Real>::meshes)
      .field("ids", &csg_domains_labeled_t<Real>::ids)
      .field("tagOffsets", &csg_domains_labeled_t<Real>::tag_offsets)
      .field("tagData", &csg_domains_labeled_t<Real>::tag_data)
      .field("faceOffsets", &csg_domains_labeled_t<Real>::face_offsets)
      .field("faceData", &csg_domains_labeled_t<Real>::face_data);

  emscripten::value_object<csg_domains_index_map_t<Real>>(
      "CsgDomainsIndexMapFloat32")
      .field("meshes", &csg_domains_index_map_t<Real>::meshes)
      .field("ids", &csg_domains_index_map_t<Real>::ids)
      .field("faceTagOffsets",
             &csg_domains_index_map_t<Real>::face_tag_offsets)
      .field("faceTagData", &csg_domains_index_map_t<Real>::face_tag_data)
      .field("faceOffsets", &csg_domains_index_map_t<Real>::face_offsets)
      .field("faceData", &csg_domains_index_map_t<Real>::face_data)
      .field("pointTagOffsets",
             &csg_domains_index_map_t<Real>::point_tag_offsets)
      .field("pointTagData", &csg_domains_index_map_t<Real>::point_tag_data)
      .field("pointOffsets", &csg_domains_index_map_t<Real>::point_offsets)
      .field("pointData", &csg_domains_index_map_t<Real>::point_data)
      .field("nOriginalPoints",
             &csg_domains_index_map_t<Real>::n_original_points)
      .field("nTags", &csg_domains_index_map_t<Real>::n_tags)
      .field("nOutputPoints",
             &csg_domains_index_map_t<Real>::n_output_points)
      .field("inclusion", &csg_domains_index_map_t<Real>::inclusion);

  // Sync
  emscripten::function("csg_graph_float32", &sync_csg_graph<Real>);
  emscripten::function("csg_mesh_float32", &sync_csg_mesh<Real>);
  emscripten::function("csg_mesh_with_labels_float32",
                       &sync_csg_mesh_with_labels<Real>);
  emscripten::function("csg_mesh_with_index_map_float32",
                       &sync_csg_mesh_with_index_map<Real>);
  emscripten::function("csg_domains_float32", &sync_csg_domains<Real>);
  emscripten::function("csg_domains_with_labels_float32",
                       &sync_csg_domains_with_labels<Real>);
  emscripten::function("csg_domains_with_index_map_float32",
                       &sync_csg_domains_with_index_map<Real>);
  emscripten::function("csg_created_points_float32",
                       &sync_csg_created_points<Real>);
  emscripten::function("csg_intersection_curves_float32",
                       &sync_csg_intersection_curves<Real>);

  // Async
  emscripten::function("dispatch_csg_graph_float32", &async_csg_graph<Real>);
  emscripten::function("dispatch_csg_intersection_curves_float32",
                       &async_csg_intersection_curves<Real>);
  emscripten::function("dispatch_csg_mesh_float32", &async_csg_mesh<Real>);
  emscripten::function("dispatch_csg_mesh_with_labels_float32",
                       &async_csg_mesh_with_labels<Real>);
  emscripten::function("dispatch_csg_mesh_with_index_map_float32",
                       &async_csg_mesh_with_index_map<Real>);
  emscripten::function("dispatch_csg_domains_float32",
                       &async_csg_domains<Real>);
  emscripten::function("dispatch_csg_domains_with_labels_float32",
                       &async_csg_domains_with_labels<Real>);
  emscripten::function("dispatch_csg_domains_with_index_map_float32",
                       &async_csg_domains_with_index_map<Real>);
}
