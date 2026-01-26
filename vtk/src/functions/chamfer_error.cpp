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
#include <trueform/geometry/chamfer_error.hpp>
#include <trueform/vtk/functions/chamfer_error.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto chamfer_error(polydata *source, polydata *target) -> float {
  auto target_with_tree = target->points() | tf::tag(target->point_tree());
  return tf::chamfer_error(source->points(), target_with_tree);
}

auto chamfer_error(std::pair<polydata *, vtkMatrix4x4 *> source,
                   polydata *target) -> float {
  auto [src_mesh, src_matrix] = source;
  tf::frame<double, 3> src_frame;
  src_frame.fill(src_matrix->GetData());
  auto target_with_tree = target->points() | tf::tag(target->point_tree());
  return tf::chamfer_error(src_mesh->points() | tf::tag(src_frame),
                           target_with_tree);
}

auto chamfer_error(polydata *source,
                   std::pair<polydata *, vtkMatrix4x4 *> target) -> float {
  auto [tgt_mesh, tgt_matrix] = target;
  tf::frame<double, 3> tgt_frame;
  tgt_frame.fill(tgt_matrix->GetData());
  auto target_with_tree =
      tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag(tgt_mesh->point_tree());
  return tf::chamfer_error(source->points(), target_with_tree);
}

auto chamfer_error(std::pair<polydata *, vtkMatrix4x4 *> source,
                   std::pair<polydata *, vtkMatrix4x4 *> target) -> float {
  auto [src_mesh, src_matrix] = source;
  auto [tgt_mesh, tgt_matrix] = target;
  tf::frame<double, 3> src_frame;
  src_frame.fill(src_matrix->GetData());
  tf::frame<double, 3> tgt_frame;
  tgt_frame.fill(tgt_matrix->GetData());
  auto target_with_tree =
      tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag(tgt_mesh->point_tree());
  return tf::chamfer_error(src_mesh->points() | tf::tag(src_frame),
                           target_with_tree);
}

} // namespace tf::vtk
