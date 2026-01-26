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
#include <trueform/core/ray_hit_info.hpp>
#include <trueform/spatial/tree_ray_info.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>

class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Result type for ray hit operations on VTK polydata.
using ray_hit_result = tf::tree_ray_info<vtkIdType, tf::ray_hit_info<float, 3>>;

/// @brief Cast a ray against a polydata mesh and return the hit point.
/// @param ray The ray to cast.
/// @param input The polydata to test against.
/// @return Ray hit result with element index and hit info including position. Convertible to bool.
auto ray_hit(tf::ray<float, 3> ray, polydata *input) -> ray_hit_result;

/// @brief Cast a ray against a polydata mesh and return the hit point.
/// @param ray The ray to cast.
/// @param input The polydata to test against.
/// @param config Ray configuration (min_t, max_t).
/// @return Ray hit result with element index and hit info including position. Convertible to bool.
auto ray_hit(tf::ray<float, 3> ray, polydata *input,
             tf::ray_config<float> config) -> ray_hit_result;

/// @brief Cast a ray against a transformed polydata mesh and return the hit point.
/// @param ray The ray to cast.
/// @param input The polydata and transform to test against.
/// @return Ray hit result with element index and hit info including position. Convertible to bool.
auto ray_hit(tf::ray<float, 3> ray, std::pair<polydata *, vtkMatrix4x4 *> input)
    -> ray_hit_result;

/// @brief Cast a ray against a transformed polydata mesh and return the hit point.
/// @param ray The ray to cast.
/// @param input The polydata and transform to test against.
/// @param config Ray configuration (min_t, max_t).
/// @return Ray hit result with element index and hit info including position. Convertible to bool.
auto ray_hit(tf::ray<float, 3> ray, std::pair<polydata *, vtkMatrix4x4 *> input,
             tf::ray_config<float> config) -> ray_hit_result;

} // namespace tf::vtk
