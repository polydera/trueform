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
#include "./decimate.hpp"
#include "./preserve_regions.hpp"
#include "./protect_vertices.hpp"
#include "./wrapper_helpers.hpp"

#include <tuple>
#include <utility>

namespace tf {

// tf::decimated builds a half-edge structure and forwards to the in-place
// driver tf::decimate, then assembles the mesh. Optional outputs come in the
// order their tags were passed (regions, protection, map):
//
//   mesh, half_edges, [face_labels], [protection_mask], [face_map, vertex_map]
//
// Decimation is pure collapse, so BOTH the face and vertex index maps are exact.

// ---- plain ----------------------------------------------------------------

template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  tf::decimate(he, points, target_proportion, config);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::pair{std::move(mesh), std::move(he)};
}

template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{});
}

// ---- index maps -> (mesh, he, face_map, vertex_map) -----------------------

template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [fim, vim] =
      tf::decimate(he, points, target_proportion, config, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(fim),
                    std::move(vim)};
}

template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   tf::return_index_map);
}

// ---- protect -> (mesh, he, protection_mask) -------------------------------

template <typename Policy, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::protect_vertices_t<Mask> protection) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto prot = tf::decimate(he, points, target_proportion, config, protection);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(prot)};
}

template <typename Policy, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::protect_vertices_t<Mask> protection) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   protection);
}

// ---- protect + maps -> (mesh, he, protection_mask, face_map, vertex_map) ---

template <typename Policy, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [prot, fim, vim] = tf::decimate(he, points, target_proportion, config,
                                       protection, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(prot),
                    std::move(fim), std::move(vim)};
}

template <typename Policy, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   protection, tf::return_index_map);
}

// ---- regions -> (mesh, he, face_labels) -----------------------------------

template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::preserve_regions_t<Range> regions) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto labels = tf::decimate(he, points, target_proportion, config, regions);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels)};
}

template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::preserve_regions_t<Range> regions) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   regions);
}

// ---- regions + maps -> (mesh, he, face_labels, face_map, vertex_map) -------

template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::preserve_regions_t<Range> regions, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, fim, vim] = tf::decimate(he, points, target_proportion, config,
                                         regions, tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(fim), std::move(vim)};
}

template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::preserve_regions_t<Range> regions, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   regions, tf::return_index_map);
}

// ---- regions + protect -> (mesh, he, face_labels, protection_mask) --------

template <typename Policy, typename Range, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::preserve_regions_t<Range> regions,
               tf::protect_vertices_t<Mask> protection) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, prot] =
      tf::decimate(he, points, target_proportion, config, regions, protection);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(prot)};
}

template <typename Policy, typename Range, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::preserve_regions_t<Range> regions,
               tf::protect_vertices_t<Mask> protection) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   regions, protection);
}

// ---- regions + protect + maps
//        -> (mesh, he, face_labels, protection_mask, face_map, vertex_map) --

template <typename Policy, typename Range, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::preserve_regions_t<Range> regions,
               tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  auto [he, points] = tf::remesh::extract_he_points(polygons);
  auto [labels, prot, fim, vim] = tf::decimate(
      he, points, target_proportion, config, regions, protection,
      tf::return_index_map);
  auto mesh = tf::remesh::make_mesh(he, std::move(points));
  return std::tuple{std::move(mesh), std::move(he), std::move(labels),
                    std::move(prot), std::move(fim), std::move(vim)};
}

template <typename Policy, typename Range, typename Mask>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::preserve_regions_t<Range> regions,
               tf::protect_vertices_t<Mask> protection, tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   regions, protection, tf::return_index_map);
}

} // namespace tf
