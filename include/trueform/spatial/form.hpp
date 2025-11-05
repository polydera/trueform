/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/coordinate_dims.hpp"
#include "../core/frame_like.hpp"
#include "../core/frame_ptr.hpp"
#include "../core/policy/frame.hpp"
#include "./policy/tree.hpp"

namespace tf {

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

template <std::size_t Dims, typename FPolicy, typename Index, typename RealT,
          typename Policy>
auto make_form(const tf::frame_like<Dims, FPolicy> &_frame,
               const tf::tree<Index, RealT, Dims> &_tree, Policy &&policy) {
  auto base =
      tf::tag_frame(tf::make_frame_ptr(_frame),
                    tf::tag_const_tree(&_tree, static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

template <std::size_t Dims, typename FPolicy, typename Index, typename RealT,
          typename Policy>
auto make_form(tf::frame_like<Dims, FPolicy> &&_frame,
               const tf::tree<Index, RealT, Dims> &_tree, Policy &&policy) {
  auto base =
      tf::tag_frame(tf::make_frame_like(_frame.transformation(),
                                        _frame.inverse_transformation()),
                    tf::tag_const_tree(&_tree, static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

template <typename Index, typename RealT, std::size_t Dims, typename Policy>
auto make_form(const tf::tree<Index, RealT, Dims> &_tree, Policy &&policy) {
  auto base = tf::tag_identity_frame<RealT, Dims>(
      tf::tag_const_tree(&_tree, static_cast<Policy &&>(policy)));
  return form<Dims, decltype(base)>{std::move(base)};
}

template <std::size_t Dims, typename Policy> auto make_form(Policy &&policy) {
  static_assert(tf::has_tree_policy<Policy>, "Form needs a tree");
  auto base = tf::tag_identity_frame<tf::coordinate_type<Policy>, Dims>(
      static_cast<Policy &&>(policy));
  return form<Dims, decltype(base)>{std::move(base)};
}

template <typename Policy> auto make_form(Policy &&policy) {
  return make_form<tf::coordinate_dims_v<Policy>>(
      static_cast<Policy &&>(policy));
}

} // namespace tf
