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
#include <trueform/cut/return_curves.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>
#include <vector>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Extract isobands (regions between scalar values).
///
/// Returns a polydata containing polygons for regions between cut values,
/// filtered by selected band indices. BandLabel cell scalars indicate which band.
///
/// @param input The input mesh with float point scalars.
/// @param scalars_name Name of the float point data array (nullptr for active scalars).
/// @param cut_values The scalar values defining band boundaries.
/// @param selected_bands Indices of bands to extract (0 = below first cut, etc.).
/// @return Polydata with BandLabel cell scalars, or nullptr on failure.
///
/// @code
/// auto bands = tf::vtk::make_isobands(mesh, "height",
///     {0.0f, 0.5f, 1.0f}, {0, 2});
/// @endcode
auto make_isobands(vtkPolyData *input, const char *scalars_name,
                   const std::vector<float> &cut_values,
                   const std::vector<int> &selected_bands)
    -> vtkSmartPointer<polydata>;

/// @brief Extract isobands with boundary curves.
///
/// @param input The input mesh with float point scalars.
/// @param scalars_name Name of the float point data array (nullptr for active scalars).
/// @param cut_values The scalar values defining band boundaries.
/// @param selected_bands Indices of bands to extract (0 = below first cut, etc.).
/// @param tag Pass tf::return_curves to get boundary curves.
/// @return Pair of (polydata with BandLabel cell scalars, curves polydata).
///
/// @code
/// auto [bands, curves] = tf::vtk::make_isobands(mesh, "height",
///     {0.0f, 0.5f, 1.0f}, {0, 2}, tf::return_curves);
/// @endcode
auto make_isobands(vtkPolyData *input, const char *scalars_name,
                   const std::vector<float> &cut_values,
                   const std::vector<int> &selected_bands,
                   tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

} // namespace tf::vtk
