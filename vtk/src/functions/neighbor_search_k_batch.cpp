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
#include <trueform/core.hpp>
#include <trueform/spatial/policy.hpp>
#include <trueform/core/algorithm/generate_offset_blocks.hpp>
#include <trueform/spatial/nearest_neighbor.hpp>
#include <trueform/spatial/neighbor_search.hpp>
#include <trueform/vtk/functions/neighbor_search_k_batch.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

namespace {

// Cascade: polygons -> segments -> points
template <typename F>
auto with_form(polydata *input, F &&f) {
  if (input->GetNumberOfPolys() > 0) {
    auto form = input->polygons() | tf::tag(input->poly_tree());
    f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(tf::make_edges(input->edges_buffer()),
                                      input->points());
    auto form = segments | tf::tag(input->segment_tree());
    f(form);
  } else {
    auto form = input->points() | tf::tag(input->point_tree());
    f(form);
  }
}

// Cascade with transform
template <typename F>
auto with_form(polydata *input, vtkMatrix4x4 *matrix, F &&f) {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  if (input->GetNumberOfPolys() > 0) {
    auto form = input->polygons() | tf::tag(input->poly_tree()) | tf::tag(frame);
    f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(tf::make_edges(input->edges_buffer()),
                                      input->points());
    auto form = segments | tf::tag(input->segment_tree()) | tf::tag(frame);
    f(form);
  } else {
    auto form = input->points() | tf::tag(input->point_tree()) | tf::tag(frame);
    f(form);
  }
}

} // namespace

// ============================================================================
// Batch kNN: Form vs Points (no radius)
// ============================================================================

auto neighbor_search_k_batch(polydata *input, vtkPoints *points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  return neighbor_search_k_batch(input, make_points(points), k);
}

auto neighbor_search_k_batch(polydata *input, points_t points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  tf::offset_block_vector<std::size_t, neighbor_result> results;

  with_form(input, [&](const auto &form) {
    tf::generate_offset_blocks(points, results, [&](auto pt, auto &out) {
      auto old_size = out.size();
      out.resize(old_size + k);
      auto knn = tf::make_nearest_neighbors(out.begin() + old_size, k);
      tf::neighbor_search(form, pt, knn);
      out.resize(old_size + knn.size());
    });
  });

  return results;
}

auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             vtkPoints *points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  return neighbor_search_k_batch(input, make_points(points), k);
}

auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             points_t points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  auto [mesh, matrix] = input;
  tf::offset_block_vector<std::size_t, neighbor_result> results;

  with_form(mesh, matrix, [&](const auto &form) {
    tf::generate_offset_blocks(points, results, [&](auto pt, auto &out) {
      auto old_size = out.size();
      out.resize(old_size + k);
      auto knn = tf::make_nearest_neighbors(out.begin() + old_size, k);
      tf::neighbor_search(form, pt, knn);
      out.resize(old_size + knn.size());
    });
  });

  return results;
}

// ============================================================================
// Batch kNN: Form vs Points (with radius)
// ============================================================================

auto neighbor_search_k_batch(polydata *input, vtkPoints *points, std::size_t k,
                             float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  return neighbor_search_k_batch(input, make_points(points), k, radius);
}

auto neighbor_search_k_batch(polydata *input, points_t points, std::size_t k,
                             float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  tf::offset_block_vector<std::size_t, neighbor_result> results;

  with_form(input, [&](const auto &form) {
    tf::generate_offset_blocks(points, results, [&](auto pt, auto &out) {
      auto old_size = out.size();
      out.resize(old_size + k);
      auto knn = tf::make_nearest_neighbors(out.begin() + old_size, k, radius);
      tf::neighbor_search(form, pt, knn);
      out.resize(old_size + knn.size());
    });
  });

  return results;
}

auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             vtkPoints *points, std::size_t k, float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  return neighbor_search_k_batch(input, make_points(points), k, radius);
}

auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             points_t points, std::size_t k, float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result> {
  auto [mesh, matrix] = input;
  tf::offset_block_vector<std::size_t, neighbor_result> results;

  with_form(mesh, matrix, [&](const auto &form) {
    tf::generate_offset_blocks(points, results, [&](auto pt, auto &out) {
      auto old_size = out.size();
      out.resize(old_size + k);
      auto knn = tf::make_nearest_neighbors(out.begin() + old_size, k, radius);
      tf::neighbor_search(form, pt, knn);
      out.resize(old_size + knn.size());
    });
  });

  return results;
}

} // namespace tf::vtk
