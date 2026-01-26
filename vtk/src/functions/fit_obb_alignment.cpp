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
#include <trueform/geometry/fit_obb_alignment.hpp>
#include <trueform/vtk/core/make_vtk_matrix.hpp>
#include <trueform/vtk/functions/fit_obb_alignment.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto fit_obb_alignment(polydata *source, polydata *target,
                       std::size_t sample_size) -> vtkSmartPointer<vtkMatrix4x4> {
  auto target_with_tree = target->points() | tf::tag(target->point_tree());
  auto T = tf::fit_obb_alignment(source->points(), target_with_tree, sample_size);
  return make_vtk_matrix(T);
}

auto fit_obb_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       polydata *target, std::size_t sample_size)
    -> vtkSmartPointer<vtkMatrix4x4> {
  auto [src_mesh, src_matrix] = source;
  tf::frame<double, 3> src_frame;
  src_frame.fill(src_matrix->GetData());
  auto target_with_tree = target->points() | tf::tag(target->point_tree());
  auto T = tf::fit_obb_alignment(src_mesh->points() | tf::tag(src_frame),
                                 target_with_tree, sample_size);
  return make_vtk_matrix(T);
}

auto fit_obb_alignment(polydata *source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       std::size_t sample_size) -> vtkSmartPointer<vtkMatrix4x4> {
  auto [tgt_mesh, tgt_matrix] = target;
  tf::frame<double, 3> tgt_frame;
  tgt_frame.fill(tgt_matrix->GetData());
  auto target_with_tree =
      tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag(tgt_mesh->point_tree());
  auto T = tf::fit_obb_alignment(source->points(), target_with_tree, sample_size);
  return make_vtk_matrix(T);
}

auto fit_obb_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       std::size_t sample_size) -> vtkSmartPointer<vtkMatrix4x4> {
  auto [src_mesh, src_matrix] = source;
  auto [tgt_mesh, tgt_matrix] = target;
  tf::frame<double, 3> src_frame;
  src_frame.fill(src_matrix->GetData());
  tf::frame<double, 3> tgt_frame;
  tgt_frame.fill(tgt_matrix->GetData());
  auto target_with_tree =
      tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag(tgt_mesh->point_tree());
  auto T = tf::fit_obb_alignment(src_mesh->points() | tf::tag(src_frame),
                                 target_with_tree, sample_size);
  return make_vtk_matrix(T);
}

} // namespace tf::vtk
