/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "./indirect_range.hpp"
#include "./unwrap.hpp"
#include <utility>

namespace tf {
/**
 * @ingroup injectors
 * @brief Type injector that adds ids to a class.
 *
 * It provides accessors for the ids.
 *
 * @tparam Range The id range
 * @tparam Base The type being augmented.
 */
template <typename Range, typename Base> struct inject_ids_t : Base {
  /**
   * @brief Constructs an instance.
   */
  inject_ids_t(const Range &_ids, const Base &base) : Base{base}, _ids{_ids} {}

  /**
   * @brief Constructs an instance.
   */
  inject_ids_t(Range &&_ids, Base &&base)
      : Base{std::move(base)}, _ids{std::move(_ids)} {}

  /**
   * @brief Returns a const reference to the ids.
   */
  auto ids() const -> const Range & { return _ids; }

  /**
   * @brief Returns a mutable reference to the ids.
   */
  auto ids() -> Range & { return _ids; }

private:
  Range _ids;
};

template <typename Range, typename Base>
struct static_size<inject_ids_t<Range, Base>> : static_size<Base> {};

namespace implementation {

template <typename Range, typename Base>
auto has_injected_ids(const tf::inject_ids_t<Range, Base> *) -> std::true_type;

template <typename Iterator, std::size_t N>
auto has_injected_ids(const tf::indirect_range<Iterator, N> *)
    -> std::true_type;

auto has_injected_ids(const void *) -> std::false_type;
} // namespace implementation

template <typename T>
inline constexpr bool has_injected_ids =
    decltype(implementation::has_injected_ids(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

/**
 * @ingroup injectors
 * @brief Constructs an `inject_ids_t` by injecting ids into a base
 */
template <typename Range, typename Base>
auto inject_ids(Range &&ids, Base &&base) {
  if constexpr (has_injected_ids<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, inject_ids_t<std::decay_t<Range>, std::decay_t<decltype(b_base)>>{
                  static_cast<Range &&>(ids), b_base});
  }
}

} // namespace tf
