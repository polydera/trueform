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
#include <string>

namespace tf::vtk {

/// @brief Read an STL file and return a polydata with cached structures.
/// @note Normals are not read from the file.
/// @param file_name Path to the STL file.
/// @return A polydata object with the mesh data.
auto read_stl(const std::string &file_name) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk
