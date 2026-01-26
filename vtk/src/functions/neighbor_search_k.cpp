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
#include <trueform/spatial/nearest_neighbor.hpp>
#include <trueform/spatial/policy.hpp>
#include <trueform/spatial/neighbor_search.hpp>
#include <trueform/vtk/functions/neighbor_search_k.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

namespace {

template <typename Knn>
auto make_results(Knn &knn) -> std::vector<neighbor_result> {
  std::vector<neighbor_result> results;
  results.reserve(knn.size());
  for (const auto &neighbor : knn) {
    results.push_back(neighbor);
  }
  return results;
}

// Cascade: polygons -> segments -> points
template <typename F>
auto with_form(polydata *input, F &&f) {
  if (input->GetNumberOfPolys() > 0) {
    auto form = input->polygons() | tf::tag(input->poly_tree());
    return f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(tf::make_edges(input->edges_buffer()),
                                      input->points());
    auto form = segments | tf::tag(input->segment_tree());
    return f(form);
  } else {
    auto form = input->points() | tf::tag(input->point_tree());
    return f(form);
  }
}

// Cascade with transform
template <typename F>
auto with_form(polydata *input, vtkMatrix4x4 *matrix, F &&f) {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  if (input->GetNumberOfPolys() > 0) {
    auto form = input->polygons() | tf::tag(input->poly_tree()) | tf::tag(frame);
    return f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(tf::make_edges(input->edges_buffer()),
                                      input->points());
    auto form = segments | tf::tag(input->segment_tree()) | tf::tag(frame);
    return f(form);
  } else {
    auto form = input->points() | tf::tag(input->point_tree()) | tf::tag(frame);
    return f(form);
  }
}

} // namespace

// ============================================================================
// kNN: Form vs Point (no radius)
// ============================================================================

auto neighbor_search_k(polydata *input, tf::point<float, 3> point,
                       std::size_t k) -> std::vector<neighbor_result> {
  return with_form(input, [&](const auto &form) {
    std::vector<neighbor_result> buffer(k);
    auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
    tf::neighbor_search(form, point, knn);
    buffer.resize(knn.size());
    return buffer;
  });
}

auto neighbor_search_k(std::pair<polydata *, vtkMatrix4x4 *> input,
                       tf::point<float, 3> point, std::size_t k)
    -> std::vector<neighbor_result> {
  auto [mesh, matrix] = input;
  return with_form(mesh, matrix, [&](const auto &form) {
    std::vector<neighbor_result> buffer(k);
    auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
    tf::neighbor_search(form, point, knn);
    buffer.resize(knn.size());
    return buffer;
  });
}

// ============================================================================
// kNN: Form vs Point (with radius)
// ============================================================================

auto neighbor_search_k(polydata *input, tf::point<float, 3> point,
                       std::size_t k, float radius)
    -> std::vector<neighbor_result> {
  return with_form(input, [&](const auto &form) {
    std::vector<neighbor_result> buffer(k);
    auto knn = tf::make_nearest_neighbors(buffer.begin(), k, radius);
    tf::neighbor_search(form, point, knn);
    buffer.resize(knn.size());
    return buffer;
  });
}

auto neighbor_search_k(std::pair<polydata *, vtkMatrix4x4 *> input,
                       tf::point<float, 3> point, std::size_t k, float radius)
    -> std::vector<neighbor_result> {
  auto [mesh, matrix] = input;
  return with_form(mesh, matrix, [&](const auto &form) {
    std::vector<neighbor_result> buffer(k);
    auto knn = tf::make_nearest_neighbors(buffer.begin(), k, radius);
    tf::neighbor_search(form, point, knn);
    buffer.resize(knn.size());
    return buffer;
  });
}

} // namespace tf::vtk
