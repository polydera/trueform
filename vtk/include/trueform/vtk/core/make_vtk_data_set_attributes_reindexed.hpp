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
#include <trueform/core/index_map.hpp>
#include <vtkSmartPointer.h>
#include <vtkType.h>

class vtkDataSetAttributes;
class vtkPointData;
class vtkCellData;

namespace tf::vtk {

/// @brief Creates a new vtkDataSetAttributes with arrays reindexed by the given map.
/// Preserves active attribute assignments (scalars, vectors, normals, etc.).
/// @param attr The source attributes.
/// @param im The index map (kept_ids specifies which tuples to keep).
/// @return A new vtkDataSetAttributes with reindexed arrays.
auto make_vtk_data_set_attributes_reindexed(
    vtkDataSetAttributes *attr, const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkDataSetAttributes>;

/// @brief Creates a new vtkPointData with arrays reindexed by the given map.
/// @param attr The source point data.
/// @param im The index map.
/// @return A new vtkPointData with reindexed arrays.
auto make_vtk_point_data_reindexed(vtkPointData *attr,
                                   const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkPointData>;

/// @brief Creates a new vtkCellData with arrays reindexed by the given map.
/// @param attr The source cell data.
/// @param im The index map.
/// @return A new vtkCellData with reindexed arrays.
auto make_vtk_cell_data_reindexed(vtkCellData *attr,
                                  const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkCellData>;

} // namespace tf::vtk
