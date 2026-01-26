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
#include <trueform/spatial/neighbor_search.hpp>
#include <trueform/spatial/policy.hpp>
#include <trueform/vtk/functions/neighbor_search.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

namespace {

// Cascade: polygons -> segments -> points
template <typename F>
auto with_form(polydata *input, F &&f) {
  if (input->GetNumberOfPolys() > 0) {
    auto form = input->polygons() | tf::tag(input->poly_tree());
    return f(form);
  } else if (input->GetNumberOfLines() > 0) {
    auto segments = tf::make_segments(
        tf::make_edges(input->edges_buffer()), input->points());
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
    auto segments = tf::make_segments(
        tf::make_edges(input->edges_buffer()), input->points());
    auto form = segments | tf::tag(input->segment_tree()) | tf::tag(frame);
    return f(form);
  } else {
    auto form = input->points() | tf::tag(input->point_tree()) | tf::tag(frame);
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
    return tf::neighbor_search(form, point);
  });
}

auto neighbor_search(polydata *input, tf::point<float, 3> point, float radius)
    -> neighbor_result {
  return with_form(input, [&](const auto &form) {
    return tf::neighbor_search(form, point, radius);
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point) -> neighbor_result {
  auto [mesh, matrix] = input;
  return with_form(mesh, matrix, [&](const auto &form) {
    return tf::neighbor_search(form, point);
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point, float radius)
    -> neighbor_result {
  auto [mesh, matrix] = input;
  return with_form(mesh, matrix, [&](const auto &form) {
    return tf::neighbor_search(form, point, radius);
  });
}

// ============================================================================
// Form vs Form
// ============================================================================

auto neighbor_search(polydata *input0, polydata *input1)
    -> neighbor_pair_result {
  return with_form(input0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1);
    });
  });
}

auto neighbor_search(polydata *input0, polydata *input1, float radius)
    -> neighbor_pair_result {
  return with_form(input0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1, radius);
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1) -> neighbor_pair_result {
  auto [mesh0, matrix0] = input0;
  return with_form(mesh0, matrix0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1);
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1, float radius)
    -> neighbor_pair_result {
  auto [mesh0, matrix0] = input0;
  return with_form(mesh0, matrix0, [&](const auto &form0) {
    return with_form(input1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1, radius);
    });
  });
}

auto neighbor_search(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> neighbor_pair_result {
  auto [mesh1, matrix1] = input1;
  return with_form(input0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1);
    });
  });
}

auto neighbor_search(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1, float radius)
    -> neighbor_pair_result {
  auto [mesh1, matrix1] = input1;
  return with_form(input0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1, radius);
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
      return tf::neighbor_search(form0, form1);
    });
  });
}

auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1, float radius)
    -> neighbor_pair_result {
  auto [mesh0, matrix0] = input0;
  auto [mesh1, matrix1] = input1;
  return with_form(mesh0, matrix0, [&, &mesh1 = mesh1, &matrix1 = matrix1](const auto &form0) {
    return with_form(mesh1, matrix1, [&](const auto &form1) {
      return tf::neighbor_search(form0, form1, radius);
    });
  });
}

} // namespace tf::vtk
