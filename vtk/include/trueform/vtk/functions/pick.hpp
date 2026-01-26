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
#include <trueform/vtk/core/polydata.hpp>
#include <vector>
#include <vtkActor.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Result of a pick operation.
struct pick_result {
  vtkActor *actor = nullptr;
  vtkIdType cell_id = -1;
  tf::point<float, 3> position;
  float t = 0;

  /// @brief Returns `true` if a valid pick was recorded.
  operator bool() const { return actor != nullptr; }
};

/// @brief Pick the closest actor along a ray.
/// @param ray The ray in world coordinates.
/// @param actors Vector of actors to test against.
/// @return Pick result with actor and hit info. Convertible to bool.
auto pick(tf::ray<float, 3> ray, std::vector<vtkActor *> &actors)
    -> pick_result;

/// @brief Pick the closest actor along a ray.
/// @param ray The ray in world coordinates.
/// @param actors Range of actor pointers to test against.
/// @return Pick result with actor and hit info. Convertible to bool.
auto pick(tf::ray<float, 3> ray,
          tf::range<vtkActor **, tf::dynamic_size> actors)
    -> pick_result;

/// @brief Pick the closest actor along a ray.
/// @param ray The ray in world coordinates.
/// @param actors Vector of smart pointer actors to test against.
/// @return Pick result with actor and hit info. Convertible to bool.
auto pick(tf::ray<float, 3> ray, std::vector<vtkSmartPointer<vtkActor>> &actors)
    -> pick_result;

/// @brief Pick the closest actor along a ray.
/// @param ray The ray in world coordinates.
/// @param actors Range of smart pointer actors to test against.
/// @return Pick result with actor and hit info. Convertible to bool.
auto pick(tf::ray<float, 3> ray,
          tf::range<vtkSmartPointer<vtkActor> *, tf::dynamic_size> actors)
    -> pick_result;

} // namespace tf::vtk
