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
#include "../iter/mapped_iterator.hpp"

namespace tf::policy {
template <typename Range, typename Dref, typename Base>
struct zipped : Base {
  using iterator = decltype(iter::make_iter_mapped(
      std::make_pair(std::declval<Range>().begin(),
                     std::declval<Base>().begin()),
      Dref{}));
  using const_iterator = decltype(iter::make_iter_mapped(
      std::make_pair(std::declval<const Range>().begin(),
                     std::declval<const Base>().begin()),
      Dref{}));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  /**
   * @brief Constructs an instance.
   */
  zipped(const Range &_zipped, const Base &base)
      : Base{base}, _zipped{_zipped} {}

  auto begin() const -> const_iterator {
    return iter::make_iter_mapped(
        std::make_pair(_zipped.begin(), Base::begin()), Dref{});
  }

  auto begin() -> iterator {
    return iter::make_iter_mapped(
        std::make_pair(_zipped.begin(), Base::begin()), Dref{});
  }

  auto end() const -> const_iterator {
    return iter::make_iter_mapped(
        std::make_pair(_zipped.end(), Base::end()), Dref{});
  }

  auto end() -> iterator {
    return iter::make_iter_mapped(
        std::make_pair(_zipped.end(), Base::end()), Dref{});
  }

  auto size() const -> size_type { return Base::size(); }

  auto empty() const -> bool { return size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(end() - 1); }

  auto back() -> reference { return *(end() - 1); }

  auto operator[](std::size_t i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](std::size_t i) -> reference { return *(begin() + i); }

protected:
  Range _zipped;
};
} // namespace tf::implementation
