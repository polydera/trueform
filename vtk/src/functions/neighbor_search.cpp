/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/core/make_buffer_for_transformed.hpp>
#include <trueform/spatial/neighbor_search.hpp>
#include <trueform/vtk/functions/neighbor_search.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

namespace {

// Helper to create result from trueform nearest_neighbor
template <typename Result>
auto make_result(const Result &result) -> neighbor_result {
  return neighbor_result{
      result.element,
      tf::sqrt(result.info.metric),
      result.info.point};
}

// Helper to create pair result from trueform nearest_neighbor_pair
template <typename Result>
auto make_pair_result(const Result &result) -> neighbor_pair_result {
  return neighbor_pair_result{
      result.elements.first,
      result.elements.second,
      tf::sqrt(result.info.metric),
      result.info.first,
      result.info.second};
}

// Cascade: polygons -> segments -> points
template <typename F>
auto with_form(polydata *input, F &&f) {
  if (input->GetNumberOfPolys() > 0) {
    auto form = tf::make_form(input->poly_tree(), input->polygons());
    return f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(
        tf::make_edges(input->edges_buffer()), input->points());
    auto form = tf::make_form(input->segment_tree(), segments);
    return f(form);
  } else {
    auto form = tf::make_form(input->point_tree(), input->points());
    return f(form);
  }
}

// Cascade with transform
template <typename F>
auto with_form(polydata *input, vtkMatrix4x4 *matrix, F &&f) {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  if (input->GetNumberOfPolys() > 0) {
    auto form = tf::make_form(frame, input->poly_tree(), input->polygons());
    return f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(
        tf::make_edges(input->edges_buffer()), input->points());
    auto form = tf::make_form(frame, input->segment_tree(), segments);
    return f(form);
  } else {
    auto form = tf::make_form(frame, input->point_tree(), input->points());
    return f(form);
  }
}

} // namespace

// ============================================================================
// Form vs Point
// ============================================================================

auto neighbor_search(polydata *input, tf::point<float, 3> point)
    -> neighbor_result {
  return with_form(input, [&](const auto &form) {
    return make_result(tf::neighbor_search(form, point));
  });
}

auto neighbor_search(polydata *input, tf::point<float, 3> point, float radius)
    -> std::optional<neighbor_result> {
  return with_form(input, [&](const auto &form) -> std::optional<neighbor_result> {
    auto result = tf::neighbor_search(form, point, radius);
    if (!result)
      return std::nullopt;
    return make_result(result);
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point) -> neighbor_result {
  auto [mesh, matrix] = input;
  return with_form(mesh, matrix, [&](const auto &form) {
    return make_result(tf::neighbor_search(form, point));
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point, float radius)
    -> std::optional<neighbor_result> {
  auto [mesh, matrix] = input;
  return with_form(mesh, matrix, [&](const auto &form) -> std::optional<neighbor_result> {
    auto result = tf::neighbor_search(form, point, radius);
    if (!result)
      return std::nullopt;
    return make_result(result);
  });
}

// ============================================================================
// Form vs Form
// ============================================================================

auto neighbor_search(polydata *input0, polydata *input1)
    -> neighbor_pair_result {
  return with_form(input0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) {
      return make_pair_result(tf::neighbor_search(form0, form1));
    });
  });
}

auto neighbor_search(polydata *input0, polydata *input1, float radius)
    -> std::optional<neighbor_pair_result> {
  return with_form(input0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) -> std::optional<neighbor_pair_result> {
      auto result = tf::neighbor_search(form0, form1, radius);
      if (!result)
        return std::nullopt;
      return make_pair_result(result);
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1) -> neighbor_pair_result {
  auto [mesh0, matrix0] = input0;
  return with_form(mesh0, matrix0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) {
      return make_pair_result(tf::neighbor_search(form0, form1));
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1, float radius)
    -> std::optional<neighbor_pair_result> {
  auto [mesh0, matrix0] = input0;
  return with_form(mesh0, matrix0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) -> std::optional<neighbor_pair_result> {
      auto result = tf::neighbor_search(form0, form1, radius);
      if (!result)
        return std::nullopt;
      return make_pair_result(result);
    });
  });
}

auto neighbor_search(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> neighbor_pair_result {
  auto [mesh1, matrix1] = input1;
  return with_form(input0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) {
      return make_pair_result(tf::neighbor_search(form0, form1));
    });
  });
}

auto neighbor_search(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1, float radius)
    -> std::optional<neighbor_pair_result> {
  auto [mesh1, matrix1] = input1;
  return with_form(input0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) -> std::optional<neighbor_pair_result> {
      auto result = tf::neighbor_search(form0, form1, radius);
      if (!result)
        return std::nullopt;
      return make_pair_result(result);
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> neighbor_pair_result {
  auto [mesh0, matrix0] = input0;
  auto [mesh1, matrix1] = input1;
  return with_form(mesh0, matrix0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) {
      return make_pair_result(tf::neighbor_search(form0, form1));
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1, float radius)
    -> std::optional<neighbor_pair_result> {
  auto [mesh0, matrix0] = input0;
  auto [mesh1, matrix1] = input1;
  return with_form(mesh0, matrix0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) -> std::optional<neighbor_pair_result> {
      auto result = tf::neighbor_search(form0, form1, radius);
      if (!result)
        return std::nullopt;
      return make_pair_result(result);
    });
  });
}

} // namespace tf::vtk
