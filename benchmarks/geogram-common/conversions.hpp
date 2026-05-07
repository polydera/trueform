/**
 * Geogram conversion utilities
 *
 * Helper functions for converting between TrueForm and Geogram data structures.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>

#include <geogram/basic/common.h>
#include <geogram/basic/numeric.h>
#include <geogram/mesh/mesh.h>

namespace benchmark {
namespace geogram {

/**
 * Populate a Geogram Mesh (3D, double coordinates) from a TrueForm
 * mesh. GEO::Mesh has a deleted copy constructor, so the caller owns
 * the destination and we fill it in place.
 *
 * @param r_polygons TrueForm polygon buffer (result from tf::read_stl etc.)
 * @param mesh       Destination Geogram mesh; cleared then populated.
 */
template <typename PolygonBuffer>
void to_geogram_mesh(const PolygonBuffer &r_polygons, GEO::Mesh &mesh) {
  mesh.clear();
  auto points = r_polygons.points();
  auto faces = r_polygons.faces();

  mesh.vertices.create_vertices(GEO::index_t(points.size()));
  for (std::size_t i = 0; i < points.size(); ++i) {
    auto pt = points[i];
    double *coords = mesh.vertices.point_ptr(GEO::index_t(i));
    coords[0] = double(pt[0]);
    coords[1] = double(pt[1]);
    coords[2] = double(pt[2]);
  }

  mesh.facets.create_triangles(GEO::index_t(faces.size()));
  for (std::size_t f = 0; f < faces.size(); ++f) {
    auto face = faces[f];
    mesh.facets.set_vertex(GEO::index_t(f), 0, GEO::index_t(face[0]));
    mesh.facets.set_vertex(GEO::index_t(f), 1, GEO::index_t(face[1]));
    mesh.facets.set_vertex(GEO::index_t(f), 2, GEO::index_t(face[2]));
  }

  mesh.facets.connect();
}

} // namespace geogram
} // namespace benchmark
