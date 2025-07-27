/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../tree.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct tag_const_tree;
template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct tag_tree;

template <typename Index, typename RealT, std::size_t Dims, typename Base>
auto has_tree(const tag_tree<Index, RealT, Dims, Base> *) -> std::true_type;

template <typename Index, typename RealT, std::size_t Dims, typename Base>
auto has_tree(const tag_const_tree<Index, RealT, Dims, Base> *)
    -> std::true_type;

auto has_tree(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_tree_policy = decltype(policy::has_tree(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct tag_tree : Base {
  using real_t = RealT;
  using index_t = Index;
  using Base::operator=;
  tag_tree(tf::tree<Index, RealT, Dims> *_tree, const Base &base)
      : Base{base}, _tree{_tree} {}

  tag_tree(tf::tree<Index, RealT, Dims> *_tree, Base &&base)
      : Base{std::move(base)}, _tree{_tree} {}

  auto tree() const -> const tf::tree<Index, RealT, Dims> & { return *_tree; }

  auto tree() -> tf::tree<Index, RealT, Dims> & { return *_tree; }

private:
  tf::tree<Index, RealT, Dims> *_tree;

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
    return tag_tree<Index, RealT, Dims, std::decay_t<T>>{val._tree,
                                                         static_cast<T &&>(t)};
  }
};

} // namespace policy

template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct static_size<policy::tag_tree<Index, RealT, Dims, Base>>
    : static_size<Base> {};

template <typename Index, typename RealT, std::size_t Dims, typename Base>
auto tag_tree(tf::tree<Index, RealT, Dims> *_tree, Base &&base) {
  if constexpr (has_tree_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_tree<Index, RealT, Dims, std::decay_t<decltype(b_base)>>{
            _tree, b_base});
  }
}

template <typename Index, typename RealT, std::size_t Dims, typename Base>
auto tag_tree(tf::tree<Index, RealT, Dims> &_tree, Base &&base) {
  return tag_tree(&_tree, static_cast<Base &&>(base));
}

namespace policy {
template <typename Index, typename RealT, std::size_t Dims> struct tag_tree_op {
  tf::tree<Index, RealT, Dims> *tree;
};

template <typename U, typename Index, typename RealT, std::size_t Dims>
auto operator|(U &&u, tag_tree_op<Index, RealT, Dims> t) {
  return tf::tag_tree(t.tree, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index, typename RealT, std::size_t Dims>
auto tag_tree(tf::tree<Index, RealT, Dims> &_tree) {
  return policy::tag_tree_op<Index, RealT, Dims>{&_tree};
}

template <typename Index, typename RealT, std::size_t Dims>
auto tag(tf::tree<Index, RealT, Dims> &_tree) {
  return policy::tag_tree_op<Index, RealT, Dims>{&_tree};
}

} // namespace tf
namespace std {
template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct tuple_size<tf::policy::tag_tree<Index, RealT, Dims, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Index, typename RealT, std::size_t Dims,
          typename Base>
struct tuple_element<I, tf::policy::tag_tree<Index, RealT, Dims, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std

namespace tf {
namespace policy {
template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct tag_const_tree : Base {

  using real_t = RealT;
  using index_t = Index;

  using Base::operator=;
  tag_const_tree(const tf::tree<Index, RealT, Dims> *_tree, const Base &base)
      : Base{base}, _tree{_tree} {}

  tag_const_tree(const tf::tree<Index, RealT, Dims> *_tree, Base &&base)
      : Base{std::move(base)}, _tree{_tree} {}

  auto tree() const -> const tf::tree<Index, RealT, Dims> & { return *_tree; }

private:
  const tf::tree<Index, RealT, Dims> *_tree;

  friend auto unwrap(const tag_const_tree &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_const_tree &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_const_tree &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_const_tree &val, T &&t) {
    return tag_const_tree<Index, RealT, Dims, std::decay_t<T>>{
        val._tree, static_cast<T &&>(t)};
  }
};
} // namespace policy
template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct static_size<policy::tag_const_tree<Index, RealT, Dims, Base>>
    : static_size<Base> {};

template <typename Index, typename RealT, std::size_t Dims, typename Base>
auto tag_const_tree(const tf::tree<Index, RealT, Dims> *_tree, Base &&base) {
  if constexpr (has_tree_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_const_tree<Index, RealT, Dims,
                               std::decay_t<decltype(b_base)>>{_tree, b_base});
  }
}

template <typename Index, typename RealT, std::size_t Dims, typename Base>
auto tag_const_tree(tf::tree<Index, RealT, Dims> &_tree, Base &&base) {
  return tag_const_tree(&_tree, static_cast<Base &&>(base));
}

namespace policy {
template <typename Index, typename RealT, std::size_t Dims>
struct tag_const_tree_op {
  const tf::tree<Index, RealT, Dims> *tree;
};

template <typename U, typename Index, typename RealT, std::size_t Dims>
auto operator|(U &&u, tag_const_tree_op<Index, RealT, Dims> t) {
  return tf::tag_const_tree(t.tree, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index, typename RealT, std::size_t Dims>
auto tag_const_tree(const tf::tree<Index, RealT, Dims> &_tree) {
  return policy::tag_const_tree_op<Index, RealT, Dims>{&_tree};
}

template <typename Index, typename RealT, std::size_t Dims>
auto tag(const tf::tree<Index, RealT, Dims> &_tree) {
  return policy::tag_const_tree_op<Index, RealT, Dims>{&_tree};
}

} // namespace tf
namespace std {
template <typename Index, typename RealT, std::size_t Dims, typename Base>
struct tuple_size<tf::policy::tag_const_tree<Index, RealT, Dims, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Index, typename RealT, std::size_t Dims,
          typename Base>
struct tuple_element<I, tf::policy::tag_const_tree<Index, RealT, Dims, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
