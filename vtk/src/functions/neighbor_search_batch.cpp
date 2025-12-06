/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/core.hpp>
#include <trueform/vtk/functions/neighbor_search.hpp>
#include <trueform/vtk/functions/neighbor_search_batch.hpp>

namespace tf::vtk {

// ============================================================================
// Batch: Form vs Points (no radius)
// ============================================================================

auto neighbor_search_batch(polydata *input, vtkPoints *points)
    -> std::vector<neighbor_result> {
  return neighbor_search_batch(input, make_points(points));
}

auto neighbor_search_batch(polydata *input, points_t points)
    -> std::vector<neighbor_result> {
  std::vector<neighbor_result> results(points.size());
  tf::parallel_apply(
      tf::zip(results, points),
      [&](auto pair) {
        auto &&[result, pt] = pair;
        result = neighbor_search(input, pt);
      },
      tf::checked);
  return results;
}

auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           vtkPoints *points) -> std::vector<neighbor_result> {
  return neighbor_search_batch(input, make_points(points));
}

auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           points_t points) -> std::vector<neighbor_result> {
  std::vector<neighbor_result> results(points.size());
  tf::parallel_apply(
      tf::zip(results, points),
      [&](auto pair) {
        auto &&[result, pt] = pair;
        result = neighbor_search(input, pt);
      },
      tf::checked);
  return results;
}

// ============================================================================
// Batch: Form vs Points (with radius)
// ============================================================================

auto neighbor_search_batch(polydata *input, vtkPoints *points, float radius)
    -> std::vector<std::optional<neighbor_result>> {
  return neighbor_search_batch(input, make_points(points), radius);
}

auto neighbor_search_batch(polydata *input, points_t points, float radius)
    -> std::vector<std::optional<neighbor_result>> {
  std::vector<std::optional<neighbor_result>> results(points.size());
  tf::parallel_apply(
      tf::zip(results, points),
      [&](auto pair) {
        auto &&[result, pt] = pair;
        result = neighbor_search(input, pt, radius);
      },
      tf::checked);
  return results;
}

auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           vtkPoints *points, float radius)
    -> std::vector<std::optional<neighbor_result>> {
  return neighbor_search_batch(input, make_points(points), radius);
}

auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           points_t points, float radius)
    -> std::vector<std::optional<neighbor_result>> {
  std::vector<std::optional<neighbor_result>> results(points.size());
  tf::parallel_apply(
      tf::zip(results, points),
      [&](auto pair) {
        auto &&[result, pt] = pair;
        result = neighbor_search(input, pt, radius);
      },
      tf::checked);
  return results;
}

} // namespace tf::vtk
