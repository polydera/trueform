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
#include "../../core/policy/unwrap.hpp"
#include "../mod_tree_like.hpp"
#include "../mod_tree.hpp"
#include "../tree_like.hpp"
#include "../tree.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename TreeViewPolicy, typename Base> struct tag_tree;
template <typename ModTreeViewPolicy, typename Base> struct tag_mod_tree;

template <typename TreeViewPolicy, typename Base>
auto has_tree(const tag_tree<TreeViewPolicy, Base> *) -> std::true_type;

template <typename ModTreeViewPolicy, typename Base>
auto has_tree(const tag_mod_tree<ModTreeViewPolicy, Base> *) -> std::true_type;

auto has_tree(const void *) -> std::false_type;
} // namespace policy

/// @ingroup spatial_policies
/// @brief Checks if a type has tree policy attached.
///
/// True if the type was wrapped with @ref tf::tag_tree() or @ref tf::tag_mod_tree().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_tree_policy = decltype(policy::has_tree(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {

template <typename TreeViewPolicy, typename Base> struct tag_tree : Base {
  using tree_policy = TreeViewPolicy;
  using index_type = typename TreeViewPolicy::index_type;

  using Base::operator=;

  tag_tree(tf::tree_like<TreeViewPolicy> _tree_view, const Base &base)
      : Base{base}, _tree_view{std::move(_tree_view)} {}

  tag_tree(tf::tree_like<TreeViewPolicy> _tree_view, Base &&base)
      : Base{std::move(base)}, _tree_view{std::move(_tree_view)} {}

  auto tree() const -> const tf::tree_like<TreeViewPolicy> & {
    return _tree_view;
  }

  auto tree() -> tf::tree_like<TreeViewPolicy> & { return _tree_view; }

private:
  tf::tree_like<TreeViewPolicy> _tree_view;

  friend auto unwrap(const tag_tree &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_tree &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_tree &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_tree &val, T &&t) {
    return tag_tree<TreeViewPolicy, std::decay_t<T>>{val._tree_view,
                                                      static_cast<T &&>(t)};
  }
};

} // namespace policy

template <typename TreeViewPolicy, typename Base>
struct static_size<policy::tag_tree<TreeViewPolicy, Base>> : static_size<Base> {
};

/// @ingroup spatial_policies
/// @brief Attaches tree data to a base type.
///
/// Creates a wrapper that carries tree information alongside the original data.
/// The result provides a `.tree()` accessor. Use with pipe syntax:
/// `data | tf::tag_tree(tree)`.
///
/// @tparam TreeViewPolicy The tree view policy type.
/// @tparam Base The base type to wrap.
/// @param _tree_view The tree view data.
/// @param base The base value to wrap.
/// @return A wrapped type with tree accessible via `.tree()`.
template <typename TreeViewPolicy, typename Base>
auto tag_tree(tf::tree_like<TreeViewPolicy> &&_tree_view, Base &&base) {
  if constexpr (has_tree_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_tree<TreeViewPolicy, std::decay_t<decltype(b_base)>>{
                  std::move(_tree_view), b_base});
  }
}

/// @overload
template <typename Index, typename BV, typename Base>
auto tag_tree(tf::tree<Index, BV> &_tree, Base &&base) {
  return tag_tree(tf::make_tree_view(_tree), static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename BV, typename Base>
auto tag_tree(const tf::tree<Index, BV> &_tree, Base &&base) {
  return tag_tree(tf::make_tree_view(_tree), static_cast<Base &&>(base));
}

template <typename Index, typename BV, typename Base>
auto tag_tree(tf::tree<Index, BV> &&_tree, Base &&base) = delete;

namespace policy {
template <typename TreeViewPolicy> struct tag_tree_op {
  tf::tree_like<TreeViewPolicy> tree_view;
};

template <typename U, typename TreeViewPolicy>
auto operator|(U &&u, tag_tree_op<TreeViewPolicy> t) {
  return tf::tag_tree(std::move(t.tree_view), static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup spatial_policies
/// @brief Creates a pipe-able tree tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// tree to a range: `data | tf::tag_tree(tree)`.
///
/// @tparam TreeViewPolicy The tree view policy type.
/// @param _tree_view The tree view data.
/// @return A tag operator for use with pipe syntax.
template <typename TreeViewPolicy>
auto tag_tree(tf::tree_like<TreeViewPolicy> &&_tree_view) {
  return policy::tag_tree_op<TreeViewPolicy>{std::move(_tree_view)};
}

/// @overload
template <typename Index, typename BV> auto tag_tree(tf::tree<Index, BV> &_tree) {
  return policy::tag_tree_op<spatial::tree_ranges<Index, BV>>{
      tf::make_tree_view(_tree)};
}

/// @overload
template <typename Index, typename BV>
auto tag_tree(const tf::tree<Index, BV> &_tree) {
  return policy::tag_tree_op<spatial::tree_ranges<Index, BV>>{
      tf::make_tree_view(_tree)};
}

template <typename Index, typename BV>
auto tag_tree(tf::tree<Index, BV> &&_tree) = delete;

/// @ingroup spatial_policies
/// @brief Creates a pipe-able tag operator for tree.
///
/// Generic overload of @ref tf::tag() that auto-detects the spatial type.
/// Equivalent to `tf::tag_tree(_tree)`.
///
/// @tparam Index The index type.
/// @tparam BV The bounding volume type.
/// @param _tree The tree structure.
/// @return A tag operator for use with pipe syntax.
template <typename Index, typename BV> auto tag(tf::tree<Index, BV> &_tree) {
  return tag_tree(_tree);
}

/// @overload
template <typename Index, typename BV>
auto tag(const tf::tree<Index, BV> &_tree) {
  return tag_tree(_tree);
}

template <typename Index, typename BV>
auto tag(tf::tree<Index, BV> &&_tree) = delete;

/// @overload
template <typename TreeViewPolicy>
auto tag(tf::tree_like<TreeViewPolicy> &_tree_view) {
  using index_type = typename TreeViewPolicy::index_type;
  using bv_type = typename TreeViewPolicy::bv_type;
  return policy::tag_tree_op<spatial::tree_ranges<index_type, bv_type>>{
      tf::make_tree_view(_tree_view)};
}

/// @overload
template <typename TreeViewPolicy>
auto tag(const tf::tree_like<TreeViewPolicy> &_tree_view) {
  using index_type = typename TreeViewPolicy::index_type;
  using bv_type = typename TreeViewPolicy::bv_type;
  return policy::tag_tree_op<spatial::tree_ranges<index_type, bv_type>>{
      tf::make_tree_view(_tree_view)};
}

/// @overload
template <typename TreeViewPolicy>
auto tag(tf::tree_like<TreeViewPolicy> &&_tree_view) {
  return tag(_tree_view);
}

// =============================================================================
// mod_tree_like support
// =============================================================================

namespace policy {

template <typename ModTreeViewPolicy, typename Base> struct tag_mod_tree : Base {
  using tree_policy = ModTreeViewPolicy;
  using index_type = typename ModTreeViewPolicy::index_type;

  using Base::operator=;

  tag_mod_tree(tf::mod_tree_like<ModTreeViewPolicy> _tree_view, const Base &base)
      : Base{base}, _tree_view{std::move(_tree_view)} {}

  tag_mod_tree(tf::mod_tree_like<ModTreeViewPolicy> _tree_view, Base &&base)
      : Base{std::move(base)}, _tree_view{std::move(_tree_view)} {}

  auto tree() const -> const tf::mod_tree_like<ModTreeViewPolicy> & {
    return _tree_view;
  }

  auto tree() -> tf::mod_tree_like<ModTreeViewPolicy> & { return _tree_view; }

private:
  tf::mod_tree_like<ModTreeViewPolicy> _tree_view;

  friend auto unwrap(const tag_mod_tree &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_mod_tree &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_mod_tree &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_mod_tree &val, T &&t) {
    return tag_mod_tree<ModTreeViewPolicy, std::decay_t<T>>{val._tree_view,
                                                             static_cast<T &&>(t)};
  }
};

} // namespace policy

template <typename ModTreeViewPolicy, typename Base>
struct static_size<policy::tag_mod_tree<ModTreeViewPolicy, Base>> : static_size<Base> {
};

/// @ingroup spatial_policies
/// @brief Attaches mod_tree data to a base type.
///
/// Creates a wrapper that carries modifiable tree information alongside the
/// original data. The result provides a `.tree()` accessor. Use with pipe syntax:
/// `data | tf::tag_mod_tree(tree)`.
///
/// @tparam ModTreeViewPolicy The mod tree view policy type.
/// @tparam Base The base type to wrap.
/// @param _tree_view The mod tree view data.
/// @param base The base value to wrap.
/// @return A wrapped type with mod_tree accessible via `.tree()`.
template <typename ModTreeViewPolicy, typename Base>
auto tag_mod_tree(tf::mod_tree_like<ModTreeViewPolicy> &&_tree_view, Base &&base) {
  if constexpr (has_tree_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_mod_tree<ModTreeViewPolicy, std::decay_t<decltype(b_base)>>{
                  std::move(_tree_view), b_base});
  }
}

/// @overload
template <typename Index, typename BV, typename Base>
auto tag_mod_tree(tf::mod_tree<Index, BV> &_tree, Base &&base) {
  return tag_mod_tree(tf::make_tree_view(_tree), static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename BV, typename Base>
auto tag_mod_tree(const tf::mod_tree<Index, BV> &_tree, Base &&base) {
  return tag_mod_tree(tf::make_tree_view(_tree), static_cast<Base &&>(base));
}

template <typename Index, typename BV, typename Base>
auto tag_mod_tree(tf::mod_tree<Index, BV> &&_tree, Base &&base) = delete;

namespace policy {
template <typename ModTreeViewPolicy> struct tag_mod_tree_op {
  tf::mod_tree_like<ModTreeViewPolicy> tree_view;
};

template <typename U, typename ModTreeViewPolicy>
auto operator|(U &&u, tag_mod_tree_op<ModTreeViewPolicy> t) {
  return tf::tag_mod_tree(std::move(t.tree_view), static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup spatial_policies
/// @brief Creates a pipe-able mod_tree tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// mod_tree to a range: `data | tf::tag_mod_tree(tree)`.
///
/// @tparam ModTreeViewPolicy The mod tree view policy type.
/// @param _tree_view The mod tree view data.
/// @return A tag operator for use with pipe syntax.
template <typename ModTreeViewPolicy>
auto tag_mod_tree(tf::mod_tree_like<ModTreeViewPolicy> &&_tree_view) {
  return policy::tag_mod_tree_op<ModTreeViewPolicy>{std::move(_tree_view)};
}

/// @overload
template <typename Index, typename BV> auto tag_mod_tree(tf::mod_tree<Index, BV> &_tree) {
  return policy::tag_mod_tree_op<spatial::mod_tree_ranges<Index, BV>>{
      tf::make_tree_view(_tree)};
}

/// @overload
template <typename Index, typename BV>
auto tag_mod_tree(const tf::mod_tree<Index, BV> &_tree) {
  return policy::tag_mod_tree_op<spatial::mod_tree_ranges<Index, BV>>{
      tf::make_tree_view(_tree)};
}

template <typename Index, typename BV>
auto tag_mod_tree(tf::mod_tree<Index, BV> &&_tree) = delete;

/// @ingroup spatial_policies
/// @brief Creates a pipe-able tag operator for mod_tree.
///
/// Generic overload of @ref tf::tag() that auto-detects the spatial type.
/// Equivalent to `tf::tag_mod_tree(_tree)`.
///
/// @tparam Index The index type.
/// @tparam BV The bounding volume type.
/// @param _tree The mod_tree structure.
/// @return A tag operator for use with pipe syntax.
template <typename Index, typename BV> auto tag(tf::mod_tree<Index, BV> &_tree) {
  return tag_mod_tree(_tree);
}

/// @overload
template <typename Index, typename BV>
auto tag(const tf::mod_tree<Index, BV> &_tree) {
  return tag_mod_tree(_tree);
}

template <typename Index, typename BV>
auto tag(tf::mod_tree<Index, BV> &&_tree) = delete;

/// @overload
template <typename ModTreeViewPolicy>
auto tag(tf::mod_tree_like<ModTreeViewPolicy> &_tree_view) {
  using index_type = typename ModTreeViewPolicy::index_type;
  using bv_type = typename ModTreeViewPolicy::bv_type;
  return policy::tag_mod_tree_op<spatial::mod_tree_ranges<index_type, bv_type>>{
      tf::make_tree_view(_tree_view)};
}

/// @overload
template <typename ModTreeViewPolicy>
auto tag(const tf::mod_tree_like<ModTreeViewPolicy> &_tree_view) {
  using index_type = typename ModTreeViewPolicy::index_type;
  using bv_type = typename ModTreeViewPolicy::bv_type;
  return policy::tag_mod_tree_op<spatial::mod_tree_ranges<index_type, bv_type>>{
      tf::make_tree_view(_tree_view)};
}

/// @overload
template <typename ModTreeViewPolicy>
auto tag(tf::mod_tree_like<ModTreeViewPolicy> &&_tree_view) {
  return tag(_tree_view);
}

} // namespace tf

namespace std {
template <typename TreeViewPolicy, typename Base>
struct tuple_size<tf::policy::tag_tree<TreeViewPolicy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename TreeViewPolicy, typename Base>
struct tuple_element<I, tf::policy::tag_tree<TreeViewPolicy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};

template <typename ModTreeViewPolicy, typename Base>
struct tuple_size<tf::policy::tag_mod_tree<ModTreeViewPolicy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename ModTreeViewPolicy, typename Base>
struct tuple_element<I, tf::policy::tag_mod_tree<ModTreeViewPolicy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
