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
#include "../../core/angle.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/unit_vector_like.hpp"
#include <cmath>

namespace tf::geometry {

/// @ingroup geometry
/// @brief The angle two adjacent faces turn through: zero where their normals
/// agree, `pi` where the pair folds back onto itself.
template <typename Policy0, typename Policy1>
auto dihedral_angle(const tf::unit_vector_like<3, Policy0> &normal0,
                    const tf::unit_vector_like<3, Policy1> &normal1)
    -> tf::rad<tf::coordinate_type<Policy0, Policy1>> {
  using T = tf::coordinate_type<Policy0, Policy1>;
  return tf::rad<T>{std::atan2(T(tf::cross(normal0, normal1).length()),
                               T(tf::dot(normal0, normal1)))};
}

} // namespace tf::geometry
