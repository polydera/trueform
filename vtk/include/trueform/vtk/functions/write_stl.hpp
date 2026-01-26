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
#include <string>
#include <utility>

class vtkPolyData;
class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Write vtkPolyData to binary STL file.
/// @param input The polydata to write (must contain triangles).
/// @param filename Output filename (.stl will be appended if not present).
/// @return true if write succeeded, false otherwise.
/// @note If cell normals are present, they are written; otherwise zero normals.
auto write_stl(vtkPolyData *input, const std::string &filename) -> bool;

/// @brief Write transformed vtkPolyData to binary STL file.
/// @param input The polydata and transformation matrix.
/// @param filename Output filename (.stl will be appended if not present).
/// @return true if write succeeded, false otherwise.
/// @note If cell normals are present, they are written; otherwise zero normals.
auto write_stl(std::pair<vtkPolyData *, vtkMatrix4x4 *> input,
               const std::string &filename) -> bool;

} // namespace tf::vtk
