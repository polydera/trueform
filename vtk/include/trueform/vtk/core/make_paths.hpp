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
#include <trueform/core.hpp>
#include <vtkCellArray.h>

namespace tf::vtk {

using paths_t = decltype(tf::make_paths(tf::make_offset_block_range(
    tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0}),
    tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0}))));

auto make_paths(vtkCellArray *cells) -> paths_t;

} // namespace tf::vtk
