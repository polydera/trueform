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
#include <trueform/remesh/remesh_config.hpp>
#include <utility>
#include <vtkSmartPointer.h>

class vtkMatrix4x4;

namespace tf::vtk {

class polydata;

/// @brief Isotropic remeshing with target edge length.
/// @param input The polydata to remesh. Assumes triangles.
/// @param config Remeshing configuration.
/// @return A new polydata with the remeshed mesh and cached half-edges.
auto isotropic_remeshed(polydata *input,
                        const tf::remesh_config<float> &config)
    -> vtkSmartPointer<polydata>;

/// @brief Isotropic remeshing with target edge length.
/// @param input The polydata to remesh. Assumes triangles.
/// @param target_length Target edge length.
/// @return A new polydata with the remeshed mesh and cached half-edges.
auto isotropic_remeshed(polydata *input, float target_length)
    -> vtkSmartPointer<polydata>;

/// @brief Isotropic remeshing with a transformation frame.
/// @param input The polydata and its transformation matrix.
/// @param config Remeshing configuration.
/// @return A new polydata with the remeshed mesh and cached half-edges.
auto isotropic_remeshed(std::pair<polydata *, vtkMatrix4x4 *> input,
                        const tf::remesh_config<float> &config)
    -> vtkSmartPointer<polydata>;

/// @brief Isotropic remeshing with a transformation frame.
/// @param input The polydata and its transformation matrix.
/// @param target_length Target edge length.
/// @return A new polydata with the remeshed mesh and cached half-edges.
auto isotropic_remeshed(std::pair<polydata *, vtkMatrix4x4 *> input,
                        float target_length)
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk
