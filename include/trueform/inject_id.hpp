/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "./static_size.hpp"
#include "./unwrap.hpp"
#include <type_traits>
#include <utility>

namespace tf {

template <typename Index, typename Base> struct inject_id_t;

namespace implementation {
template <typename Range, typename Base>
auto has_injected_id(const tf::inject_id_t<Range, Base> *) -> std::true_type;

auto has_injected_id(const void *) -> std::false_type;
} // namespace implementation

template <typename T>
inline constexpr bool has_injected_id =
    decltype(implementation::has_injected_id(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;
/**
 * @ingroup injectors
 * @brief Type injector that adds id to a class.
 *
 * It provides accessors for the id.
 *
 * @tparam Range The id range
 * @tparam Base The type being augmented.
 */
template <typename Index, typename Base> struct inject_id_t : Base {

  using Base::Base;
  using Base::operator=;
  /**
   * @brief Constructs an instance.
   */
  inject_id_t(const Index &_id, const Base &base) : Base{base}, _id{_id} {}

  /**
   * @brief Constructs an instance.
   */
  inject_id_t(Index &&_id, Base &&base)
      : Base{std::move(base)}, _id{std::move(_id)} {}

  template <typename Other>
  auto operator=(Other &&other)
      -> std::enable_if_t<has_injected_id<Other> &&
                              std::is_assignable_v<Base, Other &&>,
                          inject_id_t &> {
    Base::operator=(static_cast<Other &&>(other));
    _id = other.id();
    return *this;
  }

  /**
   * @brief Returns a const reference to the id.
   */
  auto id() const -> const Index & { return _id; }

  /**
   * @brief Returns a mutable reference to the id.
   */
  auto id() -> Index & { return _id; }

private:
  Index _id;
};

template <typename Range, typename Base>
struct static_size<inject_id_t<Range, Base>> : static_size<Base> {};

/**
 * @ingroup injectors
 * @brief Constructs an `inject_id_t` by injecting id into a base
 */
template <typename Index, typename Base>
auto inject_id(Index &&id, Base &&base) {
  if constexpr (has_injected_id<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, inject_id_t<std::decay_t<Index>, std::decay_t<decltype(b_base)>>{
                  static_cast<Index &&>(id), b_base});
  }
}

} // namespace tf
