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
#pragma once

#include "../core/frame_of.hpp"
#include "../reindex/return_index_map.hpp"
#include "./isotropic_remesh.hpp"
#include "./preserve_regions.hpp"
#include "./protect_vertices.hpp"
#include "./wrapper_helpers.hpp"

#include <tuple>
#include <utility>

namespace tf {

// tf::isotropic_remeshed builds a half-edge structure and forwards to the
// in-place driver tf::isotropic_remesh (frame-aware), then assembles the mesh.
// Optional outputs come in the order their tags were passed (regions,
// protection, map):
//
//   mesh, half_edges, [face_labels], [protection_mask], [vertex_map]
//
// The vertex_map is VERTICES ONLY (splits + flips rewrite faces). See
// tf::isotropic_remesh for the per-axis semantics.

// ---- plain ----------------------------------------------------------------

template <typename Policy>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  tf::isotropic_remesh(he, points, tf::frame_of(polygons), config);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::pair{std::move(mesh), std::move(he)};
}

template <typename Policy>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length) {
  return isotropic_remeshed(polygons,
                            tf::make_isotropic_remesh_config(target_length));
}

// ---- vertex map -> (mesh, he, vertex_map) ---------------------------------

template <typename Policy>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto vmap = tf::isotropic_remesh(he, points, tf::frame_of(polygons), config,
                                   tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(vmap)};
}

template <typename Policy>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::return_index_map_t) {
  return isotropic_remeshed(polygons,
                            tf::make_isotropic_remesh_config(target_length),
                            tf::return_index_map);
}

// ---- protect -> (mesh, he, protection_mask) -------------------------------

template <typename Policy, typename Mask>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::protect_vertices_t<Mask> protection) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto prot = tf::isotropic_remesh(he, points, tf::frame_of(polygons), config,
                                   protection);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(prot)};
}

template <typename Policy, typename Mask>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::protect_vertices_t<Mask> protection) {
  return isotropic_remeshed(
      polygons, tf::make_isotropic_remesh_config(target_length), protection);
}

// ---- protect + map -> (mesh, he, protection_mask, vertex_map) --------------

template <typename Policy, typename Mask>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [prot, vmap] = tf::isotropic_remesh(
      he, points, tf::frame_of(polygons), config, protection,
      tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(prot),
                    std::move(vmap)};
}

template <typename Policy, typename Mask>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::protect_vertices_t<Mask> protection,
                        tf::return_index_map_t) {
  return isotropic_remeshed(polygons,
                            tf::make_isotropic_remesh_config(target_length),
                            protection, tf::return_index_map);
}

// ---- regions -> (mesh, he, face_labels) -----------------------------------

template <typename Policy, typename Range>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::preserve_regions_t<Range> regions) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto labels =
      tf::isotropic_remesh(he, points, tf::frame_of(polygons), config, regions);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels)};
}

template <typename Policy, typename Range>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::preserve_regions_t<Range> regions) {
  return isotropic_remeshed(
      polygons, tf::make_isotropic_remesh_config(target_length), regions);
}

// ---- regions + map -> (mesh, he, face_labels, vertex_map) ------------------

template <typename Policy, typename Range>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::preserve_regions_t<Range> regions, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, vmap] = tf::isotropic_remesh(
      he, points, tf::frame_of(polygons), config, regions, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(vmap)};
}

template <typename Policy, typename Range>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::preserve_regions_t<Range> regions,
                        tf::return_index_map_t) {
  return isotropic_remeshed(polygons,
                            tf::make_isotropic_remesh_config(target_length),
                            regions, tf::return_index_map);
}

// ---- regions + protect -> (mesh, he, face_labels, protection_mask) --------

template <typename Policy, typename Range, typename Mask>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::preserve_regions_t<Range> regions,
    tf::protect_vertices_t<Mask> protection) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, prot] = tf::isotropic_remesh(he, points, tf::frame_of(polygons),
                                             config, regions, protection);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(prot)};
}

template <typename Policy, typename Range, typename Mask>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::preserve_regions_t<Range> regions,
                        tf::protect_vertices_t<Mask> protection) {
  return isotropic_remeshed(polygons,
                            tf::make_isotropic_remesh_config(target_length),
                            regions, protection);
}

// ---- regions + protect + map
//        -> (mesh, he, face_labels, protection_mask, vertex_map) ------------

template <typename Policy, typename Range, typename Mask>
auto isotropic_remeshed(
    const tf::polygons<Policy> &polygons,
    const tf::isotropic_remesh_config<tf::coordinate_type<Policy>> &config,
    tf::preserve_regions_t<Range> regions,
    tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, prot, vmap] =
      tf::isotropic_remesh(he, points, tf::frame_of(polygons), config, regions,
                           protection, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(prot), std::move(vmap)};
}

template <typename Policy, typename Range, typename Mask>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length,
                        tf::preserve_regions_t<Range> regions,
                        tf::protect_vertices_t<Mask> protection,
                        tf::return_index_map_t) {
  return isotropic_remeshed(polygons,
                            tf::make_isotropic_remesh_config(target_length),
                            regions, protection, tf::return_index_map);
}

} // namespace tf
