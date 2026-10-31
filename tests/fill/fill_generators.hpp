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
#include <cmath>
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

/**
 * @brief The unit sphere outside its cap above polar angle `polar_deg`, as a
 *        band of `rings` rings stepping by one rim edge. The cap's rim is
 *        vertices `[0, n)`, winding with the band, and is exactly planar
 *        unless `wobble` lifts rim vertex `k` by `wobble * sin(3 phi_k)`.
 */
template <typename Index, typename Real>
auto hole_sphere_band_mesh(int n, double polar_deg, int rings,
                           double wobble = 0.0)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  const double pi = 3.14159265358979323846;
  const double t0 = polar_deg * pi / 180.0;
  const double dt = 2.0 * pi * std::sin(t0) / n;
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  mesh.points_buffer().allocate(std::size_t(n * (rings + 1)));
  for (int j = 0; j <= rings; ++j)
    for (int k = 0; k < n; ++k) {
      const double t = t0 + j * dt, p = 2.0 * pi * k / n;
      const double lift = j == 0 ? wobble * std::sin(3.0 * p) : 0.0;
      mesh.points()[std::size_t(j * n + k)] = tf::point<Real, 3>{
          Real(std::sin(t) * std::cos(p)), Real(std::sin(t) * std::sin(p)),
          Real(std::cos(t) + lift)};
    }
  mesh.faces_buffer().allocate(std::size_t(2 * n * rings));
  std::size_t face = 0;
  for (int j = 0; j < rings; ++j)
    for (int k = 0; k < n; ++k) {
      const int inner = j * n, outer = (j + 1) * n, next = (k + 1) % n;
      mesh.faces()[face][0] = Index(inner + k);
      mesh.faces()[face][1] = Index(outer + k);
      mesh.faces()[face++][2] = Index(outer + next);
      mesh.faces()[face][0] = Index(inner + k);
      mesh.faces()[face][1] = Index(outer + next);
      mesh.faces()[face++][2] = Index(inner + next);
    }
  return mesh;
}

/**
 * @brief The one rim of `rims` that holds `vertex`, alone.
 */
template <typename Index>
auto hole_rim_holding(const tf::boundary_rims<Index> &rims, Index vertex)
    -> tf::boundary_rims<Index> {
  for (std::size_t rim = 0; rim < rims.size(); ++rim)
    for (auto at : rims.vertices[rim])
      if (at == vertex)
        return hole_turned_rim(rims, Index(rim), Index(0), false);
  return tf::boundary_rims<Index>{};
}
