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
#include <trueform/vtk/functions/write_obj.hpp>
#include <trueform/io/write_obj.hpp>
#include <trueform/vtk/core/make_polygons.hpp>
#include <vtkMatrix4x4.h>
#include <vtkPolyData.h>

namespace tf::vtk {

namespace {

auto make_frame(vtkMatrix4x4 *matrix) -> tf::frame<double, 3> {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  return frame;
}

} // namespace

auto write_obj(vtkPolyData *input, const std::string &filename) -> bool {
  if (!input) {
    return false;
  }

  auto polygons = make_polygons(input);
  return tf::write_obj(polygons, filename);
}

auto write_obj(std::pair<vtkPolyData *, vtkMatrix4x4 *> input,
               const std::string &filename) -> bool {
  auto *poly = input.first;
  auto *matrix = input.second;

  if (!poly) {
    return false;
  }

  auto polygons = make_polygons(poly);

  if (matrix) {
    auto frame = make_frame(matrix);
    return tf::write_obj(polygons | tf::tag(frame), filename);
  } else {
    return tf::write_obj(polygons, filename);
  }
}

} // namespace tf::vtk
