/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../tree_like.hpp"
#include "../tree.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename TreeViewPolicy, typename Base> struct tag_tree;

template <typename TreeViewPolicy, typename Base>
auto has_tree(const tag_tree<TreeViewPolicy, Base> *) -> std::true_type;

auto has_tree(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_tree_policy = decltype(policy::has_tree(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {

template <typename TreeViewPolicy, typename Base> struct tag_tree : Base {
  using tree_policy = TreeViewPolicy;
  using index_t = typename TreeViewPolicy::index_type;
  using real_t = typename TreeViewPolicy::coordinate_type;

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

// tag_tree from tree_like (view)
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

// tag_tree from owning tree (creates view)
template <typename Index, typename BV, typename Base>
auto tag_tree(tf::tree<Index, BV> &_tree, Base &&base) {
  return tag_tree(tf::make_tree_view(_tree), static_cast<Base &&>(base));
}

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

template <typename TreeViewPolicy>
auto tag_tree(tf::tree_like<TreeViewPolicy> &&_tree_view) {
  return policy::tag_tree_op<TreeViewPolicy>{std::move(_tree_view)};
}

template <typename Index, typename BV> auto tag_tree(tf::tree<Index, BV> &_tree) {
  return policy::tag_tree_op<spatial::tree_ranges<Index, BV>>{
      tf::make_tree_view(_tree)};
}

template <typename Index, typename BV>
auto tag_tree(const tf::tree<Index, BV> &_tree) {
  return policy::tag_tree_op<spatial::tree_ranges<Index, BV>>{
      tf::make_tree_view(_tree)};
}

template <typename Index, typename BV>
auto tag_tree(tf::tree<Index, BV> &&_tree) = delete;

template <typename Index, typename BV> auto tag(tf::tree<Index, BV> &_tree) {
  return tag_tree(_tree);
}

template <typename Index, typename BV>
auto tag(const tf::tree<Index, BV> &_tree) {
  return tag_tree(_tree);
}

template <typename Index, typename BV>
auto tag(tf::tree<Index, BV> &&_tree) = delete;

template <typename TreeViewPolicy>
auto tag(tf::tree_like<TreeViewPolicy> &_tree_view) {
  using index_type = typename TreeViewPolicy::index_type;
  using bv_type = typename TreeViewPolicy::bv_type;
  return policy::tag_tree_op<spatial::tree_ranges<index_type, bv_type>>{
      tf::make_tree_view(_tree_view)};
}

template <typename TreeViewPolicy>
auto tag(const tf::tree_like<TreeViewPolicy> &_tree_view) {
  using index_type = typename TreeViewPolicy::index_type;
  using bv_type = typename TreeViewPolicy::bv_type;
  return policy::tag_tree_op<spatial::tree_ranges<index_type, bv_type>>{
      tf::make_tree_view(_tree_view)};
}

template <typename TreeViewPolicy>
auto tag(tf::tree_like<TreeViewPolicy> &&_tree_view) {
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
} // namespace std
