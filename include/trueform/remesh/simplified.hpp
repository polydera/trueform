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

#include "../reindex/return_index_map.hpp"
#include "./preserve_regions.hpp"
#include "./protect_vertices.hpp"
#include "./simplify.hpp"
#include "./wrapper_helpers.hpp"

#include <tuple>
#include <utility>

namespace tf {

// tf::simplified builds a half-edge structure and forwards to the in-place
// driver tf::simplify, then assembles the output mesh. The optional outputs
// come in the order their tags were passed (regions, protection, map):
//
//   mesh, half_edges, [face_labels], [protection_mask], [vertex_map]
//
// The vertex_map is VERTICES ONLY (the flip pass rewrites faces). See
// tf::simplify for the per-axis semantics.

// ---- plain ----------------------------------------------------------------

template <typename Policy>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  tf::simplify(he, points, config);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::pair{std::move(mesh), std::move(he)};
}

template <typename Policy>
auto simplified(const tf::polygons<Policy> &polygons) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{});
}

// ---- vertex map -> (mesh, he, vertex_map) ---------------------------------

template <typename Policy>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto vmap = tf::simplify(he, points, config, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(vmap)};
}

template <typename Policy>
auto simplified(const tf::polygons<Policy> &polygons, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, tf::return_index_map);
}

// ---- protect -> (mesh, he, protection_mask) -------------------------------

template <typename Policy, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::protect_vertices_t<Mask> protection) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto prot = tf::simplify(he, points, config, protection);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(prot)};
}

template <typename Policy, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::protect_vertices_t<Mask> protection) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, protection);
}

// ---- protect + map -> (mesh, he, protection_mask, vertex_map) --------------

template <typename Policy, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [prot, vmap] =
      tf::simplify(he, points, config, protection, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(prot),
                    std::move(vmap)};
}

template <typename Policy, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, protection,
                    tf::return_index_map);
}

// ---- regions -> (mesh, he, face_labels) -----------------------------------

template <typename Policy, typename Range>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::preserve_regions_t<Range> regions) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto labels = tf::simplify(he, points, config, regions);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels)};
}

template <typename Policy, typename Range>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::preserve_regions_t<Range> regions) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, regions);
}

// ---- regions + map -> (mesh, he, face_labels, vertex_map) ------------------

template <typename Policy, typename Range>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::preserve_regions_t<Range> regions, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, vmap] =
      tf::simplify(he, points, config, regions, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(vmap)};
}

template <typename Policy, typename Range>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::preserve_regions_t<Range> regions, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, regions,
                    tf::return_index_map);
}

// ---- regions + protect -> (mesh, he, face_labels, protection_mask) --------

template <typename Policy, typename Range, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::preserve_regions_t<Range> regions,
                tf::protect_vertices_t<Mask> protection) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, prot] = tf::simplify(he, points, config, regions, protection);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(prot)};
}

template <typename Policy, typename Range, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::preserve_regions_t<Range> regions,
                tf::protect_vertices_t<Mask> protection) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, regions, protection);
}

// ---- regions + protect + map
//        -> (mesh, he, face_labels, protection_mask, vertex_map) ------------

template <typename Policy, typename Range, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::preserve_regions_t<Range> regions,
                tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, prot, vmap] = tf::simplify(he, points, config, regions,
                                           protection, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(prot), std::move(vmap)};
}

template <typename Policy, typename Range, typename Mask>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::preserve_regions_t<Range> regions,
                tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, regions, protection,
                    tf::return_index_map);
}

} // namespace tf
