/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/polydata.hpp>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Extract both boundary and non-manifold edges from a mesh.
///
/// Returns a polydata containing line cells representing edges that are
/// either boundary (belong to one face) or non-manifold (belong to more
/// than two faces). Cell scalars "EdgeType" indicate type: 0=boundary, 1=non-manifold.
/// The output shares points with the input.
///
/// @param input The input mesh (must be tf::vtk::polydata).
/// @return Polydata with non-simple edges as lines and EdgeType cell scalars.
///
/// @code
/// auto edges = tf::vtk::make_non_simple_edges(mesh);
/// std::cout << "Found " << edges->GetNumberOfLines() << " non-simple edges\n";
/// @endcode
auto make_non_simple_edges(polydata *input) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk
