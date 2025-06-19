/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./buffer.hpp"
#include "./implementation/blocked_iterator.hpp"

namespace tf {
template <typename T, std::size_t BlockSize> class blocked_buffer {
public:
  using iterator = implementation::iter::blocked_iterator<T *, BlockSize>;
  using const_iterator =
      implementation::iter::blocked_iterator<const T *, BlockSize>;
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using size_type = typename std::iterator_traits<iterator>::difference_type;

  auto begin() const -> const_iterator {
    return implementation::iter::make_blocked_iterator<BlockSize>(
        _data.begin());
  }

  auto begin() -> iterator {
    return implementation::iter::make_blocked_iterator<BlockSize>(
        _data.begin());
  }

  auto end() const -> const_iterator {
    return implementation::iter::make_blocked_iterator<BlockSize>(_data.end());
  }

  auto end() -> iterator {
    return implementation::iter::make_blocked_iterator<BlockSize>(_data.end());
  }

  auto size() const -> size_type { return _data.size(); }

  auto empty() const -> bool { return size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(end() - 1); }

  auto back() -> reference { return *(end() - 1); }

  auto operator[](std::size_t i) const -> const reference {
    return *(begin() + i);
  }

  auto operator[](std::size_t i) -> reference { return *(begin() + i); }

  auto data_buffer() const -> const tf::buffer<T> & { return _data; }
  auto data_buffer() -> tf::buffer<T> & { return _data; }

  auto clear() { _data.clear(); }

private:
  tf::buffer<T> _data;
};

} // namespace tf
