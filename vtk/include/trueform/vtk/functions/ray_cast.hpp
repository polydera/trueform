/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>

class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Result of a ray cast operation.
struct ray_cast_result {
  vtkIdType cell_id;
  float t;
};

/// @brief Cast a ray against a polydata mesh.
/// @param ray The ray to cast.
/// @param input The polydata to test against.
/// @return Ray cast result if hit, nullopt otherwise.
auto ray_cast(tf::ray<float, 3> ray, polydata *input)
    -> std::optional<ray_cast_result>;

/// @brief Cast a ray against a polydata mesh with ray config.
/// @param ray The ray to cast.
/// @param input The polydata to test against.
/// @param config Ray configuration (min_t, max_t).
/// @return Ray cast result if hit, nullopt otherwise.
auto ray_cast(tf::ray<float, 3> ray, polydata *input,
              tf::ray_config<float> config) -> std::optional<ray_cast_result>;

/// @brief Cast a ray against a transformed polydata mesh.
/// @param ray The ray to cast.
/// @param input The polydata and transform to test against.
/// @return Ray cast result if hit, nullopt otherwise.
auto ray_cast(tf::ray<float, 3> ray,
              std::pair<polydata *, vtkMatrix4x4 *> input)
    -> std::optional<ray_cast_result>;

/// @brief Cast a ray against a transformed polydata mesh with ray config.
/// @param ray The ray to cast.
/// @param input The polydata and transform to test against.
/// @param config Ray configuration (min_t, max_t).
/// @return Ray cast result if hit, nullopt otherwise.
auto ray_cast(tf::ray<float, 3> ray,
              std::pair<polydata *, vtkMatrix4x4 *> input,
              tf::ray_config<float> config) -> std::optional<ray_cast_result>;

} // namespace tf::vtk
