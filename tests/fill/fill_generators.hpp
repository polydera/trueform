/**
 * @file fill_generators.hpp
 * @brief Meshes and rims the hole-filling suite builds its cases from
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/reindex/by_mask.hpp>
#include <trueform/topology/boundary_rims.hpp>
#include <cstddef>
#include <vector>

/**
 * @brief The cone over a closed rim: one triangle per rim edge, all meeting
 *        at one apex below it. The rim is the mesh's only boundary and every
 *        rim edge is carried by a triangle of its own.
 */
template <typename Index, typename Real>
auto hole_cone_mesh(const std::vector<tf::point<Real, 3>> &rim,
                    const tf::point<Real, 3> &apex)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  const Index n = Index(rim.size());
  mesh.points_buffer().allocate(rim.size() + 1);
  for (Index k = 0; k < n; ++k)
    mesh.points()[std::size_t(k)] = rim[std::size_t(k)];
  mesh.points()[std::size_t(n)] = apex;
  mesh.faces_buffer().allocate(rim.size());
  for (Index k = 0; k < n; ++k) {
    mesh.faces()[std::size_t(k)][0] = k;
    mesh.faces()[std::size_t(k)][1] = Index(k + 1 == n ? 0 : k + 1);
    mesh.faces()[std::size_t(k)][2] = n;
  }
  return mesh;
}

/**
 * @brief The mesh with the named faces removed.
 */
template <typename Mesh>
auto hole_punched_mesh(const Mesh &mesh, const std::vector<std::size_t> &drop) {
  tf::buffer<bool> mask;
  for (std::size_t face = 0; face < mesh.faces().size(); ++face)
    mask.push_back(true);
  for (auto face : drop)
    mask[face] = false;
  return tf::reindexed_by_mask(mesh.polygons(), tf::make_range(mask));
}

/**
 * @brief One rim, its vertices and carrying faces stated directly.
 */
template <typename Index>
auto hole_single_rim(const std::vector<Index> &vertices,
                     const std::vector<Index> &faces)
    -> tf::boundary_rims<Index> {
  tf::boundary_rims<Index> rims;
  rims.vertices.offsets_buffer().push_back(0);
  for (auto vertex : vertices)
    rims.vertices.data_buffer().push_back(vertex);
  rims.vertices.offsets_buffer().push_back(Index(vertices.size()));
  rims.faces.offsets_buffer().push_back(0);
  for (auto face : faces)
    rims.faces.data_buffer().push_back(face);
  rims.faces.offsets_buffer().push_back(Index(faces.size()));
  rims.closed.push_back(true);
  return rims;
}

/**
 * @brief The rim walked from another starting vertex, optionally backwards.
 */
template <typename Index>
auto hole_turned_rim(const tf::boundary_rims<Index> &rims, Index rim,
                     Index root, bool reverse) -> tf::boundary_rims<Index> {
  const auto vertices = rims.vertices[std::size_t(rim)];
  const auto faces = rims.faces[std::size_t(rim)];
  const Index n = Index(vertices.size());
  std::vector<Index> turned_vertices, turned_faces;
  for (Index k = 0; k < n; ++k) {
    const Index at = reverse ? Index((root + n - k) % n) : Index((root + k) % n);
    turned_vertices.push_back(vertices[std::size_t(at)]);
    turned_faces.push_back(
        faces[std::size_t(reverse ? Index((at + n - 1) % n) : at)]);
  }
  return hole_single_rim(turned_vertices, turned_faces);
}
