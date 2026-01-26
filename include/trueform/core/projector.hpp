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
#include "./basis.hpp"
#include "./coordinate_type.hpp"
#include "./point.hpp"
#include "./policy/normal.hpp"
#include "./policy/plane.hpp"
#include "./polygon.hpp"
#include "./vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A wrapper around a callable object used to project a point into a
/// lower-dimensional space.
///
/// `projector<F>` wraps a callable object `F` and provides a projection
/// operator for `point_like` inputs. This allows geometric structures like
/// planes or polygons to define custom projection behavior for operations such
/// as containment or distance calculation.
///
/// @tparam F A callable type implementing `operator()(point_like)` returning a
/// projected point.
template <typename Policy> struct projector : Policy {
  /// @brief Constructs a projector from a given function object.
  /// @param f The projection function or functor to wrap.
  projector(Policy f) : Policy{std::move(f)} {}
};

namespace core {

/// @ingroup core_primitives
/// @brief Creates a `projector` from a given callable object.
///
/// Convenience factory for wrapping a lambda or functor into a `projector`.
///
/// @tparam F The callable type.
/// @param f The callable to wrap.
/// @return A `projector` wrapping the provided callable.
template <typename F> auto make_projector(F &&f) {
  return projector<std::decay_t<F>>{static_cast<F &&>(f)};
}
} // namespace core

/// @ingroup core_primitives
/// @brief Creates an identity projector that returns the input unchanged.
///
/// Useful when no projection is needed but an interface expects a projector.
///
/// @return A projector that returns its input unchanged.
inline auto make_identity_projector() {
  return core::make_projector([](const auto &x) -> const auto & { return x; });
}

/// @ingroup core_primitives
/// @brief Creates a simple 2D projection from a 3D normal vector.
///
/// This function chooses the two coordinate axes most orthogonal to the normal,
/// and returns a projector that maps a 3D vector to a 2D vector in that plane.
///
/// The largest component of the normal is dropped to avoid distortion.
///
/// @tparam T The scalar type of the normal.
/// @param normal The normal vector (assumed normalized).
/// @return A projector mapping 3D points to 2D using orthogonal axes.
template <typename T>
auto make_simple_projector(const vector_like<3, T> &normal) {
  std::array<int, 2> ids{0, 1};
  auto max = std::abs(normal[0]);
  int k = 0;
  for (int i = 1; i < 3; i++) {
    auto tmp = std::abs(normal[i]);
    if (tmp > max) {
      k = i;
      max = tmp;
    }
  }
  if (k == 1) {
    ids[0] = 2;
    ids[1] = 0;
  } else if (k == 0) {
    ids[0] = 1;
    ids[1] = 2;
  }

  if (normal[k] < 0)
    std::swap(ids[0], ids[1]);

  return core::make_projector([x = ids[0], y = ids[1]](const auto &pt) {
    return tf::point<tf::coordinate_type<T>, 2>{pt[x], pt[y]};
  });
}

template <std::size_t Dims, typename Policy>
auto make_simple_projector(const tf::polygon<Dims, Policy> &poly) {
  auto tagged = tf::tag_normal(poly);
  return make_simple_projector(tagged.normal());
}

template <std::size_t Dims, typename Policy>
auto make_projector(const tf::plane_like<Dims, Policy> &plane) {
  auto _basis = tf::make_basis(plane);
  struct projector_t {
    decltype(_basis) basis;
    auto operator()(tf::point<tf::coordinate_type<Policy>, Dims> point) const {
      return tf::point<tf::coordinate_type<Policy>, 2>{
          tf::dot(basis[0], point), tf::dot(basis[1], point)};
    }
  };
  return core::make_projector(projector_t{_basis});
}

template <std::size_t Dims, typename Policy>
auto make_projector(const tf::polygon<Dims, Policy> &poly) {
  auto tagged = tf::tag_plane(poly);
  return make_projector(tagged.plane());
}
} // namespace tf
