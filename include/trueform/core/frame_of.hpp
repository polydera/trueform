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
#include "./coordinate_dims.hpp"
#include "./form.hpp"
#include "./frame.hpp"
#include "./points.hpp"
#include "./policy/frame.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./transformation.hpp"
#include "./unit_vectors.hpp"
#include "./vectors.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief Get the coordinate frame of a form.
///
/// Returns the tagged frame if present, otherwise returns identity frame.
///
/// @tparam Dims The spatial dimensionality.
/// @tparam Policy The form's policy type.
/// @param f The form to query.
/// @return The frame or @ref tf::identity_frame if none tagged.
template <std::size_t Dims, typename Policy>
auto frame_of(const tf::form<Dims, Policy> &f) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return f.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>, Dims>{};
  }
}

/// @ingroup core_properties
/// @brief Get the transformation matrix of a form's frame.
/// @overload
template <std::size_t Dims, typename Policy>
auto transformation_of(const tf::form<Dims, Policy> &f) -> decltype(auto) {
  return frame_of(f).transformation();
}

/// @ingroup core_properties
/// @brief Get the inverse transformation matrix of a form's frame.
/// @overload
template <std::size_t Dims, typename Policy>
auto inverse_transformation_of(const tf::form<Dims, Policy> &f)
    -> decltype(auto) {
  return frame_of(f).inverse_transformation();
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a point set.
///
/// Returns the tagged frame if present, otherwise returns identity frame.
///
/// @tparam Policy The points policy type.
/// @param t The point set to query.
/// @return The frame or @ref tf::identity_frame if none tagged.
template <typename Policy>
auto frame_of(const tf::points<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a polygon mesh.
/// @overload
template <typename Policy>
auto frame_of(const tf::polygons<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<decltype(t.points()[0])>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a vector set.
/// @overload
template <typename Policy>
auto frame_of(const tf::vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a unit vector set.
/// @overload
template <typename Policy>
auto frame_of(const tf::unit_vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a segment set.
/// @overload
template <typename Policy>
auto frame_of(const tf::segments<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<decltype(t.points()[0])>>{};
  }
}

// =============================================================================
// transformation_of - convenience for frame_of(x).transformation()
// =============================================================================

/// @ingroup core_properties
/// @brief Get the transformation matrix of a point set's frame.
///
/// Returns the transformation from local to world coordinates.
/// If no frame is tagged, returns identity matrix.
///
/// @tparam Policy The points policy type.
/// @param t The point set to query.
/// @return The transformation matrix.
template <typename Policy>
auto transformation_of(const tf::points<Policy> &t) -> decltype(auto) {
  return frame_of(t).transformation();
}

/// @ingroup core_properties
/// @brief Get the transformation matrix of a polygon mesh's frame.
/// @overload
template <typename Policy>
auto transformation_of(const tf::polygons<Policy> &t) -> decltype(auto) {
  return frame_of(t).transformation();
}

/// @ingroup core_properties
/// @brief Get the transformation matrix of a vector set's frame.
/// @overload
template <typename Policy>
auto transformation_of(const tf::vectors<Policy> &t) -> decltype(auto) {
  return frame_of(t).transformation();
}

/// @ingroup core_properties
/// @brief Get the transformation matrix of a unit vector set's frame.
/// @overload
template <typename Policy>
auto transformation_of(const tf::unit_vectors<Policy> &t) -> decltype(auto) {
  return frame_of(t).transformation();
}

/// @ingroup core_properties
/// @brief Get the transformation matrix of a segment set's frame.
/// @overload
template <typename Policy>
auto transformation_of(const tf::segments<Policy> &t) -> decltype(auto) {
  return frame_of(t).transformation();
}

// =============================================================================
// inverse_transformation_of - convenience for
// frame_of(x).inverse_transformation()
// =============================================================================

/// @ingroup core_properties
/// @brief Get the inverse transformation matrix of a point set's frame.
///
/// Returns the transformation from world to local coordinates.
/// If no frame is tagged, returns identity matrix.
///
/// @tparam Policy The points policy type.
/// @param t The point set to query.
/// @return The inverse transformation matrix.
template <typename Policy>
auto inverse_transformation_of(const tf::points<Policy> &t) -> decltype(auto) {
  return frame_of(t).inverse_transformation();
}

/// @ingroup core_properties
/// @brief Get the inverse transformation matrix of a polygon mesh's frame.
/// @overload
template <typename Policy>
auto inverse_transformation_of(const tf::polygons<Policy> &t)
    -> decltype(auto) {
  return frame_of(t).inverse_transformation();
}

/// @ingroup core_properties
/// @brief Get the inverse transformation matrix of a vector set's frame.
/// @overload
template <typename Policy>
auto inverse_transformation_of(const tf::vectors<Policy> &t) -> decltype(auto) {
  return frame_of(t).inverse_transformation();
}

/// @ingroup core_properties
/// @brief Get the inverse transformation matrix of a unit vector set's frame.
/// @overload
template <typename Policy>
auto inverse_transformation_of(const tf::unit_vectors<Policy> &t)
    -> decltype(auto) {
  return frame_of(t).inverse_transformation();
}

/// @ingroup core_properties
/// @brief Get the inverse transformation matrix of a segment set's frame.
/// @overload
template <typename Policy>
auto inverse_transformation_of(const tf::segments<Policy> &t)
    -> decltype(auto) {
  return frame_of(t).inverse_transformation();
}

// =============================================================================
// concrete_frame_of - always returns a concrete frame (not identity_frame type)
// =============================================================================

/// @ingroup core_properties
/// @brief Get a concrete frame from a form (not identity_frame type).
///
/// If the form has a tagged frame, returns it. Otherwise returns a
/// default-constructed frame (which represents identity but as a concrete
/// type).
///
/// @tparam Dims The spatial dimensionality.
/// @tparam Policy The form's policy type.
/// @param f The form to query.
/// @return A concrete frame object.
template <std::size_t Dims, typename Policy>
auto concrete_frame_of(const tf::form<Dims, Policy> &f) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return f.frame();
  else {
    return tf::make_identity_frame<tf::coordinate_type<Policy>, Dims>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete frame from a point set (not identity_frame type).
/// @overload
template <typename Policy>
auto concrete_frame_of(const tf::points<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::make_identity_frame<tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete frame from a polygon mesh (not identity_frame type).
/// @overload
template <typename Policy>
auto concrete_frame_of(const tf::polygons<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::make_identity_frame<tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete frame from a vector set (not identity_frame type).
/// @overload
template <typename Policy>
auto concrete_frame_of(const tf::vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::make_identity_frame<tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete frame from a unit vector set (not identity_frame
/// type).
/// @overload
template <typename Policy>
auto concrete_frame_of(const tf::unit_vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::make_identity_frame<tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete frame from a segment set (not identity_frame type).
/// @overload
template <typename Policy>
auto concrete_frame_of(const tf::segments<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::make_identity_frame<tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>();
  }
}

// =============================================================================
// concrete_transformation_of - always returns a concrete transformation
// =============================================================================

/// @ingroup core_properties
/// @brief Get a concrete transformation from a form's frame.
/// @overload
template <std::size_t Dims, typename Policy>
auto concrete_transformation_of(const tf::form<Dims, Policy> &f)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return f.frame().transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            Dims>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete transformation from a point set's frame.
/// @overload
template <typename Policy>
auto concrete_transformation_of(const tf::points<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete transformation from a polygon mesh's frame.
/// @overload
template <typename Policy>
auto concrete_transformation_of(const tf::polygons<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete transformation from a vector set's frame.
/// @overload
template <typename Policy>
auto concrete_transformation_of(const tf::vectors<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete transformation from a unit vector set's frame.
/// @overload
template <typename Policy>
auto concrete_transformation_of(const tf::unit_vectors<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete transformation from a segment set's frame.
/// @overload
template <typename Policy>
auto concrete_transformation_of(const tf::segments<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

// =============================================================================
// concrete_inverse_transformation_of - always returns a concrete inverse
// =============================================================================

/// @ingroup core_properties
/// @brief Get a concrete inverse transformation from a form's frame.
/// @overload
template <std::size_t Dims, typename Policy>
auto concrete_inverse_transformation_of(const tf::form<Dims, Policy> &f)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return f.frame().inverse_transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            Dims>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete inverse transformation from a point set's frame.
/// @overload
template <typename Policy>
auto concrete_inverse_transformation_of(const tf::points<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().inverse_transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete inverse transformation from a polygon mesh's frame.
/// @overload
template <typename Policy>
auto concrete_inverse_transformation_of(const tf::polygons<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().inverse_transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete inverse transformation from a vector set's frame.
/// @overload
template <typename Policy>
auto concrete_inverse_transformation_of(const tf::vectors<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().inverse_transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete inverse transformation from a unit vector set's frame.
/// @overload
template <typename Policy>
auto concrete_inverse_transformation_of(const tf::unit_vectors<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().inverse_transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

/// @ingroup core_properties
/// @brief Get a concrete inverse transformation from a segment set's frame.
/// @overload
template <typename Policy>
auto concrete_inverse_transformation_of(const tf::segments<Policy> &t)
    -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame().inverse_transformation();
  else {
    return tf::make_identity_transformation<tf::coordinate_type<Policy>,
                                            tf::coordinate_dims_v<Policy>>();
  }
}

} // namespace tf
