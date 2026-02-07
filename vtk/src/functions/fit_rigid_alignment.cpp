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
#include <trueform/geometry/fit_rigid_alignment.hpp>
#include <trueform/vtk/core/make_vtk_matrix.hpp>
#include <trueform/vtk/functions/fit_rigid_alignment.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto fit_rigid_alignment(polydata *source, polydata *target)
    -> vtkSmartPointer<vtkMatrix4x4> {
  auto src_normals = source->point_normals();
  auto tgt_normals = target->point_normals();

  if (src_normals.size() > 0 && tgt_normals.size() > 0) {
    // Both have normals: point-to-plane with normal weighting
    auto src = source->points() | tf::tag_normals(src_normals);
    auto tgt = target->points() | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  } else if (tgt_normals.size() > 0) {
    // Target has normals: point-to-plane
    auto tgt = target->points() | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(source->points(), tgt));
  } else {
    // No normals: point-to-point
    return make_vtk_matrix(
        tf::fit_rigid_alignment(source->points(), target->points()));
  }
}

auto fit_rigid_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                         polydata *target) -> vtkSmartPointer<vtkMatrix4x4> {
  auto [src_mesh, src_matrix] = source;
  tf::frame<double, 3> src_frame;
  src_frame.fill(src_matrix->GetData());

  auto src_normals = src_mesh->point_normals();
  auto tgt_normals = target->point_normals();

  if (src_normals.size() > 0 && tgt_normals.size() > 0) {
    auto src =
        src_mesh->points() | tf::tag(src_frame) | tf::tag_normals(src_normals);
    auto tgt = target->points() | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  } else if (tgt_normals.size() > 0) {
    auto src = src_mesh->points() | tf::tag(src_frame);
    auto tgt = target->points() | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  } else {
    auto src = src_mesh->points() | tf::tag(src_frame);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, target->points()));
  }
}

auto fit_rigid_alignment(polydata *source,
                         std::pair<polydata *, vtkMatrix4x4 *> target)
    -> vtkSmartPointer<vtkMatrix4x4> {
  auto [tgt_mesh, tgt_matrix] = target;
  tf::frame<double, 3> tgt_frame;
  tgt_frame.fill(tgt_matrix->GetData());

  auto src_normals = source->point_normals();
  auto tgt_normals = tgt_mesh->point_normals();

  if (src_normals.size() > 0 && tgt_normals.size() > 0) {
    auto src = source->points() | tf::tag_normals(src_normals);
    auto tgt =
        tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  } else if (tgt_normals.size() > 0) {
    auto tgt =
        tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(source->points(), tgt));
  } else {
    auto tgt = tgt_mesh->points() | tf::tag(tgt_frame);
    return make_vtk_matrix(tf::fit_rigid_alignment(source->points(), tgt));
  }
}

auto fit_rigid_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                         std::pair<polydata *, vtkMatrix4x4 *> target)
    -> vtkSmartPointer<vtkMatrix4x4> {
  auto [src_mesh, src_matrix] = source;
  auto [tgt_mesh, tgt_matrix] = target;
  tf::frame<double, 3> src_frame;
  src_frame.fill(src_matrix->GetData());
  tf::frame<double, 3> tgt_frame;
  tgt_frame.fill(tgt_matrix->GetData());

  auto src_normals = src_mesh->point_normals();
  auto tgt_normals = tgt_mesh->point_normals();

  if (src_normals.size() > 0 && tgt_normals.size() > 0) {
    auto src =
        src_mesh->points() | tf::tag(src_frame) | tf::tag_normals(src_normals);
    auto tgt =
        tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  } else if (tgt_normals.size() > 0) {
    auto src = src_mesh->points() | tf::tag(src_frame);
    auto tgt =
        tgt_mesh->points() | tf::tag(tgt_frame) | tf::tag_normals(tgt_normals);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  } else {
    auto src = src_mesh->points() | tf::tag(src_frame);
    auto tgt = tgt_mesh->points() | tf::tag(tgt_frame);
    return make_vtk_matrix(tf::fit_rigid_alignment(src, tgt));
  }
}

} // namespace tf::vtk
