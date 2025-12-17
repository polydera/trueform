/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_frame.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto make_frame(vtkMatrix4x4 *matrix) -> tf::frame<double, 3> {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  return frame;
}

template <typename T>
auto fill_frame(tf::frame<T, 3> &frame, vtkMatrix4x4 *matrix) -> void {
  frame.fill(matrix->GetData());
}

template auto fill_frame<float>(tf::frame<float, 3> &, vtkMatrix4x4 *) -> void;
template auto fill_frame<double>(tf::frame<double, 3> &, vtkMatrix4x4 *) -> void;

} // namespace tf::vtk
