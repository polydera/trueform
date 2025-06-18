#pragma once
#include "trueform/blocked_range.hpp"
#include "trueform/points.hpp"
#include "trueform/polygons.hpp"
#include "vtkPolyData.h"

inline auto get_points(vtkPoints * points) {
  auto ptr =
      static_cast<float *>(points->GetData()->GetVoidPointer(0));
  return tf::make_points<3>(
      tf::make_range(ptr, 3 * points->GetNumberOfPoints()));
}

inline auto get_points(vtkPolyData *poly) {
  auto ptr =
      static_cast<float *>(poly->GetPoints()->GetData()->GetVoidPointer(0));
  return tf::make_points<3>(
      tf::make_range(ptr, 3 * poly->GetNumberOfPoints()));
}

#ifdef VTK_CELL_ARRAY_V2
#include "trueform/blocked_range.hpp"
inline auto get_triangle_faces(vtkCellArray *_cell_array) {
  return tf::make_blocked_range<3>(tf::make_range(
      static_cast<vtkIdType *>(
          _cell_array->GetConnectivityArray()->GetVoidPointer(0)),
      3 * _cell_array->GetNumberOfCells()));
}
#else
#include "trueform/tag_blocked_range.hpp"
inline auto get_triangle_faces(vtkCellArray *_cell_array) {
  return tf::make_tag_blocked_range<3>(
      tf::make_range(_cell_array->GetPointer(),
                     _cell_array->GetNumberOfConnectivityEntries()));
}
#endif

inline auto get_triangle_faces(vtkPolyData *poly) {
  return get_triangle_faces(poly->GetPolys());
}

inline auto get_triangles(vtkPolyData *poly) {
  return tf::make_polygons(get_triangle_faces(poly), get_points(poly));
}
