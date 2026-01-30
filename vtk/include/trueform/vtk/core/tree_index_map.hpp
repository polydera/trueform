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
#include <trueform/core/range.hpp>
#include <trueform/spatial/tree_index_map.hpp>
#include <vtkType.h>

namespace tf::vtk {

/// @brief Tree index map type for VTK polydata incremental tree updates.
///
/// Used with update_poly_tree() for remapping scenarios (e.g., after boolean
/// operations). The forward mapping f() maps old IDs to new IDs (or sentinel
/// if removed), and dirty_ids() contains IDs of new/modified elements.
using tree_index_map_t = tf::tree_index_map<
    tf::range<vtkIdType *, tf::dynamic_size>,
    tf::range<vtkIdType *, tf::dynamic_size>>;

} // namespace tf::vtk
