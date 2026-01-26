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
#include <trueform/topology/connectivity_type.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Label connected components in a mesh.
///
/// Returns a polydata with ComponentLabel cell scalars indicating which
/// connected component each face belongs to.
///
/// @param input The input mesh (must be tf::vtk::polydata).
/// @param type The connectivity type to use.
/// @return Pair of (polydata with ComponentLabel cell scalars, number of components).
///
/// @code
/// auto [labeled, n] = tf::vtk::make_connected_components(mesh, tf::connectivity_type::edge);
/// std::cout << "Found " << n << " components\n";
/// @endcode
auto make_connected_components(polydata *input, tf::connectivity_type type)
    -> std::pair<vtkSmartPointer<polydata>, int>;

} // namespace tf::vtk
