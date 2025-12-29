/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (www.trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once

#include "../core/coordinate_dims.hpp"
#include "../core/frame_like.hpp"
#include "../core/frame_ptr.hpp"
#include "../core/policy/frame.hpp"
#include "./mod_tree_like.hpp"
#include "./policy/tree.hpp"

namespace tf {

/// @ingroup spatial_structures
/// @brief A positioned spatial tree with an associated coordinate frame.
///
/// A form combines a spatial tree with a @ref tf::frame, enabling
/// queries on primitives that are transformed relative to a local coordinate
/// system. This is essential for rigid body simulations and collision detection
/// where objects move through space.
///
/// Forms are created using @ref tf::make_form() factory functions, which accept
/// combinations of frames, trees (or mod_trees), and policy decorators.
///
/// @tparam Dims The spatial dimensionality (typically 2 or 3).
/// @tparam Policy The underlying policy chain (frame + tree + user policies).
template <std::size_t Dims, typename Policy> struct form : public Policy {
  form(Policy &&policy) : Policy{std::move(policy)} {}
  form(const Policy &policy) : Policy{policy} {}

  friend auto unwrap(const form &seg) -> decltype(auto) {
    return unwrap(static_cast<const Policy &>(seg));
  }

  friend auto unwrap(form &seg) -> decltype(auto) {
    return unwrap(static_cast<Policy &>(seg));
  }

  friend auto unwrap(form &&seg) -> decltype(auto) {
    return unwrap(static_cast<Policy &&>(seg));
  }

  template <typename T> friend auto wrap_like(const form &f, T &&t) {
    auto base = wrap_like(static_cast<const Policy &>(f), static_cast<T &&>(t));
    return form<Dims, decltype(base)>{std::move(base)};
  }

  template <typename T> friend auto wrap_like(form &f, T &&t) {
    auto base = wrap_like(static_cast<Policy &>(f), static_cast<T &&>(t));
    return form<Dims, decltype(base)>{std::move(base)};
  }

  template <typename T> friend auto wrap_like(form &&f, T &&t) {
    auto base = wrap_like(static_cast<Policy &&>(f), static_cast<T &&>(t));
    return form<Dims, decltype(base)>{std::move(base)};
  }
};

/// @ingroup spatial_structures
/// @brief Create a form from a frame, tree, and policy.
///
/// The frame is stored by pointer, so the original frame must outlive the form.
///
/// @tparam Dims The spatial dimensionality.
/// @tparam FPolicy Frame policy type.
/// @tparam TreePolicy Tree policy type.
/// @tparam Policy User policy type.
/// @param _frame The coordinate frame for positioning.
/// @param _tree The spatial tree for acceleration.
/// @param policy User-provided policy decorators (e.g., primitives range).
/// @return A form wrapping the frame, tree, and policy.
template <std::size_t Dims, typename FPolicy, typename TreePolicy,
          typename Policy>
auto make_form(const tf::frame_like<Dims, FPolicy> &_frame,
               const tf::tree_like<TreePolicy> &_tree, Policy &&policy) {
  static_assert(Dims == TreePolicy::coordinate_dims::value,
                "Frame and tree dimension mismatch");
  auto tree_view = tf::make_tree_view(_tree);
  auto base =
      tf::tag_frame(tf::make_frame_ptr(_frame),
                    tf::tag_tree(std::move(tree_view),
                                 static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

/// @ingroup spatial_structures
/// @brief Create a form from a moved frame, tree, and policy.
///
/// The frame is moved into the form, which owns a copy of the transformation.
///
/// @tparam Dims The spatial dimensionality.
/// @tparam FPolicy Frame policy type.
/// @tparam TreePolicy Tree policy type.
/// @tparam Policy User policy type.
/// @param _frame The coordinate frame (moved).
/// @param _tree The spatial tree for acceleration.
/// @param policy User-provided policy decorators.
/// @return A form owning the frame data.
template <std::size_t Dims, typename FPolicy, typename TreePolicy,
          typename Policy>
auto make_form(tf::frame_like<Dims, FPolicy> &&_frame,
               const tf::tree_like<TreePolicy> &_tree, Policy &&policy) {
  static_assert(Dims == TreePolicy::coordinate_dims::value,
                "Frame and tree dimension mismatch");
  auto tree_view = tf::make_tree_view(_tree);
  auto base =
      tf::tag_frame(tf::make_frame_like(_frame.transformation(),
                                        _frame.inverse_transformation()),
                    tf::tag_tree(std::move(tree_view),
                                 static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

/// @ingroup spatial_structures
/// @brief Create a form from a tree with an identity frame.
///
/// Useful when the tree is already in world coordinates.
///
/// @tparam TreePolicy Tree policy type.
/// @tparam Policy User policy type.
/// @param _tree The spatial tree for acceleration.
/// @param policy User-provided policy decorators.
/// @return A form with identity transformation.
template <typename TreePolicy, typename Policy>
auto make_form(const tf::tree_like<TreePolicy> &_tree, Policy &&policy) {
  constexpr std::size_t Dims = TreePolicy::coordinate_dims::value;
  using real_t = typename TreePolicy::coordinate_type;
  auto tree_view = tf::make_tree_view(_tree);
  auto base = tf::tag_identity_frame<real_t, Dims>(
      tf::tag_tree(std::move(tree_view), static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

/// @ingroup spatial_structures
/// @brief Create a form from a policy that already contains a tree.
///
/// @tparam Dims The spatial dimensionality.
/// @tparam Policy Policy type (must have tree attached via tag_tree).
/// @param policy Policy with tree already attached.
/// @return A form with identity transformation.
template <std::size_t Dims, typename Policy> auto make_form(Policy &&policy) {
  static_assert(tf::has_tree_policy<Policy>, "Form needs a tree");
  auto base = tf::tag_identity_frame<tf::coordinate_type<Policy>, Dims>(
      static_cast<Policy &&>(policy));
  return form<Dims, decltype(base)>{std::move(base)};
}

/// @ingroup spatial_structures
/// @brief Create a form from a policy, deducing dimensionality.
///
/// @tparam Policy Policy type (must have tree attached).
/// @param policy Policy with tree already attached.
/// @return A form with identity transformation.
template <typename Policy> auto make_form(Policy &&policy) {
  return make_form<tf::coordinate_dims_v<Policy>>(
      static_cast<Policy &&>(policy));
}

// =============================================================================
// mod_tree_like overloads
// =============================================================================

/// @ingroup spatial_structures
/// @brief Create a form from a frame, mod_tree, and policy.
///
/// @tparam Dims The spatial dimensionality.
/// @tparam FPolicy Frame policy type.
/// @tparam ModTreePolicy Mod tree policy type.
/// @tparam Policy User policy type.
/// @param _frame The coordinate frame for positioning.
/// @param _tree The dynamic spatial tree for acceleration.
/// @param policy User-provided policy decorators.
/// @return A form wrapping the frame, mod_tree, and policy.
template <std::size_t Dims, typename FPolicy, typename ModTreePolicy,
          typename Policy>
auto make_form(const tf::frame_like<Dims, FPolicy> &_frame,
               const tf::mod_tree_like<ModTreePolicy> &_tree, Policy &&policy) {
  static_assert(Dims == ModTreePolicy::coordinate_dims::value,
                "Frame and tree dimension mismatch");
  auto tree_view = tf::make_tree_view(_tree);
  auto base =
      tf::tag_frame(tf::make_frame_ptr(_frame),
                    tf::tag_mod_tree(std::move(tree_view),
                                     static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

/// @ingroup spatial_structures
/// @brief Create a form from a moved frame, mod_tree, and policy.
///
/// @tparam Dims The spatial dimensionality.
/// @tparam FPolicy Frame policy type.
/// @tparam ModTreePolicy Mod tree policy type.
/// @tparam Policy User policy type.
/// @param _frame The coordinate frame (moved).
/// @param _tree The dynamic spatial tree for acceleration.
/// @param policy User-provided policy decorators.
/// @return A form owning the frame data.
template <std::size_t Dims, typename FPolicy, typename ModTreePolicy,
          typename Policy>
auto make_form(tf::frame_like<Dims, FPolicy> &&_frame,
               const tf::mod_tree_like<ModTreePolicy> &_tree, Policy &&policy) {
  static_assert(Dims == ModTreePolicy::coordinate_dims::value,
                "Frame and tree dimension mismatch");
  auto tree_view = tf::make_tree_view(_tree);
  auto base =
      tf::tag_frame(tf::make_frame_like(_frame.transformation(),
                                        _frame.inverse_transformation()),
                    tf::tag_mod_tree(std::move(tree_view),
                                     static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

/// @ingroup spatial_structures
/// @brief Create a form from a mod_tree with an identity frame.
///
/// @tparam ModTreePolicy Mod tree policy type.
/// @tparam Policy User policy type.
/// @param _tree The dynamic spatial tree for acceleration.
/// @param policy User-provided policy decorators.
/// @return A form with identity transformation.
template <typename ModTreePolicy, typename Policy>
auto make_form(const tf::mod_tree_like<ModTreePolicy> &_tree, Policy &&policy) {
  constexpr std::size_t Dims = ModTreePolicy::coordinate_dims::value;
  using real_t = typename ModTreePolicy::coordinate_type;
  auto tree_view = tf::make_tree_view(_tree);
  auto base = tf::tag_identity_frame<real_t, Dims>(
      tf::tag_mod_tree(std::move(tree_view), static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

} // namespace tf
