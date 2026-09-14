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
#pragma once
#include <vtkAOSDataArrayTemplate.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>

#include <trueform/core/algorithm/parallel_contains.hpp>
#include <trueform/core/checked.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/slide_range.hpp>

#include <vtkType.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tf::vtk {

/// What a polydata must already be for its memory to be read where it lies.
/// Each of these is a fact about the SOURCE, refused here rather than
/// reinterpreted: a borrowed pointer that means something else is not an
/// error a later read can find.
namespace source {

template <typename Real>
auto require_point_type(vtkPolyData *polydata) -> void {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "a mesh source reads float or double points");
  auto *points = polydata ? polydata->GetPoints() : nullptr;
  if (!points)
    return;
  const auto wanted = std::is_same_v<Real, float> ? VTK_FLOAT : VTK_DOUBLE;
  if (points->GetDataType() != wanted)
    throw std::invalid_argument(
        std::string("mesh source: points are ") +
        points->GetData()->GetDataTypeAsString() + ", not " +
        (std::is_same_v<Real, float> ? "float" : "double"));
}

/// Trueform reads a point as Dims consecutive reals, which is what an array of
/// structures is and what an array of arrays is not. This is the one producer
/// of that pointer, so nothing reads the coordinates without having asked.
template <typename Real>
auto interleaved_points(vtkPolyData *polydata) -> Real * {
  auto *points = polydata ? polydata->GetPoints() : nullptr;
  if (!points || points->GetNumberOfPoints() == 0)
    return nullptr;
  auto *coordinates =
      vtkAOSDataArrayTemplate<Real>::SafeDownCast(points->GetData());
  if (!coordinates)
    throw std::invalid_argument(
        "mesh source: points are not stored one point at a time");
  return coordinates->GetPointer(0);
}

/// The facade's contract is that a cell array stores 64-bit ids, which is what
/// VTK is built with everywhere it is used. A source that says otherwise is
/// not read at another width, it is refused: its ids are not where a vtkIdType
/// reader would look. This is the one producer of that refusal, and it is
/// asked of a cell array whatever it holds, because an empty one filled later
/// is the same array.
inline auto require_cell_id_width(vtkCellArray *cells) -> void {
  if (!cells)
    return;
  if (cells->IsStorage64Bit() != (sizeof(vtkIdType) == 8))
    throw std::invalid_argument(std::string("mesh source: cells store ") +
                                (cells->IsStorage64Bit() ? "64" : "32") +
                                "-bit ids, and vtkIdType is " +
                                std::to_string(sizeof(vtkIdType) * 8) + "-bit");
}

inline auto cell_offset_count(vtkCellArray *cells) -> std::size_t {
  if (!cells || !cells->GetOffsetsArray())
    return 0;
  return static_cast<std::size_t>(
      cells->GetOffsetsArray()->GetNumberOfValues());
}

inline auto cell_connectivity_count(vtkCellArray *cells) -> std::size_t {
  if (!cells || !cells->GetConnectivityArray())
    return 0;
  return static_cast<std::size_t>(
      cells->GetConnectivityArray()->GetNumberOfValues());
}

/// The one producer of both pointers, so nothing reads connectivity without
/// having asked for the width first. A source with no cells has none, which is
/// the empty reading a mesh_geometry already states as no offsets at all.
/// The ids are read at the INDEX THE MATRIX CARRIES, not at `vtkIdType`: the
/// two have the same width and signedness, which is the same object layout,
/// but on an LP64 target they are DISTINCT types and only one of them names a
/// carrier this archive was built with.
template <typename Index> auto cell_offsets(vtkCellArray *cells) -> Index * {
  require_cell_id_width(cells);
  if (cell_offset_count(cells) == 0)
    return nullptr;
  return static_cast<Index *>(cells->GetOffsetsArray()->GetVoidPointer(0));
}

template <typename Index>
auto cell_connectivity(vtkCellArray *cells) -> Index * {
  require_cell_id_width(cells);
  if (cell_connectivity_count(cells) == 0)
    return nullptr;
  return static_cast<Index *>(cells->GetConnectivityArray()->GetVoidPointer(0));
}

/// A face is at least a triangle, and the polys are the only cells a mesh is
/// read from — strips and verts state something else.
inline auto require_face_arity(vtkPolyData *polydata) -> void {
  if (!polydata)
    return;
  if (polydata->GetNumberOfStrips() != 0)
    throw std::invalid_argument(
        "mesh source: triangle strips are not faces; run vtkTriangleFilter");
  auto *cells = polydata->GetPolys();
  const auto *offsets = cell_offsets<vtkIdType>(cells);
  const auto count = cell_offset_count(cells);
  if (count < 2)
    return;
  const auto refused = tf::parallel_contains(
      tf::make_slide_range<2>(tf::make_range(offsets, count)),
      [](const auto &block) { return block[1] - block[0] < 3; }, tf::checked);
  if (refused)
    throw std::invalid_argument("mesh source: a face is at least three points");
}

/// The four facts a borrow depends on. The fifth — that every id names a point
/// the source has — is the structure's, checked once per reading against the
/// source's own modification time.
template <typename Real> auto require_readable(vtkPolyData *polydata) -> void {
  require_point_type<Real>(polydata);
  static_cast<void>(interleaved_points<Real>(polydata));
  require_cell_id_width(polydata ? polydata->GetPolys() : nullptr);
  require_face_arity(polydata);
}

} // namespace source

} // namespace tf::vtk
