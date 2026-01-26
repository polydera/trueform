/**
 * VTK conversion utilities
 *
 * Helper functions for converting between TrueForm and VTK data structures.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>

namespace benchmark {
namespace vtk {

/**
 * Convert TrueForm polygons_buffer to VTK PolyData.
 */
template <typename Index>
vtkSmartPointer<vtkPolyData> to_vtk_polydata(
    const tf::polygons_buffer<Index, float, 3, 3>& polys) {

    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(polys.points().size());
    auto pts_ptr = static_cast<float*>(points->GetData()->GetVoidPointer(0));
    for (std::size_t i = 0; i < polys.points().size(); ++i) {
        pts_ptr[3*i + 0] = polys.points()[i][0];
        pts_ptr[3*i + 1] = polys.points()[i][1];
        pts_ptr[3*i + 2] = polys.points()[i][2];
    }

    auto cells = vtkSmartPointer<vtkCellArray>::New();
    for (std::size_t i = 0; i < polys.faces().size(); ++i) {
        vtkIdType ids[3] = {
            static_cast<vtkIdType>(polys.faces()[i][0]),
            static_cast<vtkIdType>(polys.faces()[i][1]),
            static_cast<vtkIdType>(polys.faces()[i][2])
        };
        cells->InsertNextCell(3, ids);
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetPolys(cells);
    return polydata;
}

}  // namespace vtk
}  // namespace benchmark
