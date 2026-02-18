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
#include <trueform/remesh/length_collapse_config.hpp>
#include <utility>
#include <vtkSmartPointer.h>

class vtkMatrix4x4;

namespace tf::vtk {

class polydata;

/// @brief Collapse edges shorter than a threshold.
/// @param input The polydata to process. Assumes triangles.
/// @param min_length Minimum edge length; shorter edges are collapsed.
/// @param config Collapse configuration.
/// @return A new polydata with short edges collapsed and cached half-edges.
auto collapsed_short_edges(polydata *input, float min_length,
                           const tf::length_collapse_config<float> &config = {})
    -> vtkSmartPointer<polydata>;

/// @brief Collapse edges shorter than a threshold with a transformation frame.
/// @param input The polydata and its transformation matrix.
/// @param min_length Minimum edge length; shorter edges are collapsed.
/// @param config Collapse configuration.
/// @return A new polydata with short edges collapsed and cached half-edges.
auto collapsed_short_edges(std::pair<polydata *, vtkMatrix4x4 *> input,
                           float min_length,
                           const tf::length_collapse_config<float> &config = {})
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk
