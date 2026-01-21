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
#include "../core/sqrt.hpp"
#include "../core/constants.hpp"
#include "../core/transformation.hpp"
#include "../core/vector.hpp"
#include "./random.hpp"

namespace tf {
/// @ingroup random
/// @brief Generates a random 3D rigid transformation matrix with optional
/// translation.
///
/// This function returns a randomly generated 3D transformation of type
/// @ref tf::transformation. The transformation includes a uniformly sampled
/// random rotation matrix and an optional translation vector. The rotation is
/// sampled uniformly over SO(3) using the method of uniform sampling from a
/// quaternion-like approach (see "Uniform Random Rotations" by Ken Shoemake,
/// 1992).
///
/// @tparam T The scalar type, typically `float` or `double`.
/// @param translation Optional translation vector. Defaults to the zero vector.
/// @return A 3D transformation matrix of type @ref tf::transformation<T, 3>.
///
/// @see @ref tf::random
/// @see @ref tf::transformation
template <typename U>
auto random_transformation(tf::vector_like<3, U> translation)
    -> tf::transformation<tf::coordinate_type<U>, 3> {
  using T = tf::coordinate_type<U>;
  T x0_rand = tf::random<T>(T(0), T(1));
  T x1_rand = tf::random<T>(T(0), T(1));
  T x2_rand = tf::random<T>(T(0), T(1));


  T theta = x0_rand * tf::two_pi<T>; // Rotation about the pole (Z)
  T phi = x1_rand * tf::two_pi<T>;   // For direction of pole deflection
  T z_val =
      x2_rand * static_cast<T>(2.0); // For magnitude of pole deflection [0,2]

  T r_sqrt_z = tf::sqrt(z_val);
  T Vx = std::sin(phi) * r_sqrt_z;
  T Vy = std::cos(phi) * r_sqrt_z;
  T Vz = tf::sqrt(static_cast<T>(2.0) - z_val);

  T st = std::sin(theta);
  T ct = std::cos(theta);
  T Sx = Vx * ct - Vy * st;
  T Sy = Vx * st + Vy * ct;

  // Matrix elements for the 3x3 rotation part
  T m00 = Vx * Sx - ct;
  T m01 = Vx * Sy - st;
  T m02 = Vx * Vz;

  T m10 = Vy * Sx + st;
  T m11 = Vy * Sy - ct;
  T m12 = Vy * Vz;

  T m20 = Vz * Sx;
  T m21 = Vz * Sy;
  T m22 = static_cast<T>(1.0) - z_val;

  return {{{{m00, m01, m02, translation[0]},
            {m10, m11, m12, translation[1]},
            {m20, m21, m22, translation[2]}}}};
}

template <typename U>
auto random_transformation(tf::vector_like<2, U> translation)
    -> tf::transformation<tf::coordinate_type<U>, 2> {
  using T = tf::coordinate_type<U>;

  T theta = tf::random<T>(T(0), tf::two_pi<T>);

  T ct = std::cos(theta);
  T st = std::sin(theta);

  return {{{{ct, -st, translation[0]}, {st, ct, translation[1]}}}};
}

template <typename T, std::size_t Dims> auto random_transformation() {
  if constexpr (Dims == 2)
    return random_transformation(tf::vector<T, 2>{T(0), T(0)});
  else if constexpr (Dims == 3)
    return random_transformation(tf::vector<T, 3>{T(0), T(0), T(0)});
}
} // namespace tf
