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
#include <trueform/core/ray_cast_info.hpp>
#include <trueform/spatial/tree_ray_info.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>

class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Result type for ray cast operations on VTK polydata.
using ray_cast_result = tf::tree_ray_info<vtkIdType, tf::ray_cast_info<float>>;

/// @brief Cast a ray against a polydata mesh.
/// @param ray The ray to cast.
/// @param input The polydata to test against.
/// @return Ray cast result with element index and hit info. Convertible to bool.
auto ray_cast(tf::ray<float, 3> ray, polydata *input) -> ray_cast_result;

/// @brief Cast a ray against a polydata mesh with ray config.
/// @param ray The ray to cast.
/// @param input The polydata to test against.
/// @param config Ray configuration (min_t, max_t).
/// @return Ray cast result with element index and hit info. Convertible to bool.
auto ray_cast(tf::ray<float, 3> ray, polydata *input,
              tf::ray_config<float> config) -> ray_cast_result;

/// @brief Cast a ray against a transformed polydata mesh.
/// @param ray The ray to cast.
/// @param input The polydata and transform to test against.
/// @return Ray cast result with element index and hit info. Convertible to bool.
auto ray_cast(tf::ray<float, 3> ray,
              std::pair<polydata *, vtkMatrix4x4 *> input) -> ray_cast_result;

/// @brief Cast a ray against a transformed polydata mesh with ray config.
/// @param ray The ray to cast.
/// @param input The polydata and transform to test against.
/// @param config Ray configuration (min_t, max_t).
/// @return Ray cast result with element index and hit info. Convertible to bool.
auto ray_cast(tf::ray<float, 3> ray,
              std::pair<polydata *, vtkMatrix4x4 *> input,
              tf::ray_config<float> config) -> ray_cast_result;

} // namespace tf::vtk
