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
#include <trueform/vtk/core/polydata.hpp>
#include <vector>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Split a labeled polydata into separate components.
///
/// Takes a vtkPolyData with vtkIdType cell scalars (e.g., from connected_components)
/// and returns a vector of separate polydata objects, one per unique label.
///
/// @param input vtkPolyData with vtkIdType cell scalars as labels.
/// @return Pair of (vector of polydata, vector of labels).
///
/// @code
/// vtkNew<tf::vtk::connected_components> cc;
/// cc->SetInputConnection(adapter->GetOutputPort());
/// cc->Update();
///
/// auto [components, labels] = tf::vtk::split_into_components(cc->GetOutput());
/// for (auto& component : components) {
///     // Process each component
/// }
/// @endcode
auto split_into_components(vtkPolyData *input)
    -> std::pair<std::vector<vtkSmartPointer<polydata>>, std::vector<vtkIdType>>;

/// @brief Split a polydata by a named cell data array.
///
/// @param input vtkPolyData with cell data.
/// @param array_name Name of the vtkIdType cell data array to split by.
/// @return Pair of (vector of polydata, vector of labels).
auto split_into_components(vtkPolyData *input, const char *array_name)
    -> std::pair<std::vector<vtkSmartPointer<polydata>>, std::vector<vtkIdType>>;

} // namespace tf::vtk
