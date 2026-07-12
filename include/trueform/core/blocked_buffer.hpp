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
#include "./buffer.hpp"
#include "./iter/blocked_iterator.hpp"
#include "./static_size.hpp"

namespace tf {

/// @ingroup core_buffers
/// @brief A buffer of fixed-size blocks.
///
/// Stores elements in contiguous blocks of `BlockSize` elements each.
/// Iteration yields @ref tf::range views over each block.
///
/// @tparam T The scalar element type.
/// @tparam BlockSize Number of elements per block.
template <typename T, std::size_t BlockSize> class blocked_buffer {
public:
  using iterator = iter::blocked_iterator<T *, BlockSize>;
  using const_iterator = iter::blocked_iterator<const T *, BlockSize>;
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using size_type = typename std::iterator_traits<iterator>::difference_type;

  blocked_buffer() = default;
  blocked_buffer(const tf::buffer<T> &_data) : _data{_data} {}
  blocked_buffer(tf::buffer<T> &&_data) : _data{std::move(_data)} {}

  template <typename Iterator>
  auto push_back(const tf::range<Iterator, BlockSize> &r) {
    for (std::size_t i = 0; i < BlockSize; ++i)
      _data.push_back(r[i]);
  }

  auto push_back(const std::array<T, BlockSize> &r) {
    for (std::size_t i = 0; i < BlockSize; ++i)
      _data.push_back(r[i]);
  }

  template <typename... Ts>
  auto emplace_back(Ts &&...ts)
      -> std::enable_if_t<(sizeof...(Ts) == BlockSize), void> {
    (_data.push_back(ts), ...);
  }

  auto erase(iterator from, iterator to) {
    _data.erase(from.base_iter(), to.base_iter());
  }

  auto reserve(std::size_t n) { _data.reserve(n * BlockSize); }

  auto allocate(std::size_t n) { _data.allocate(n * BlockSize); }

  auto reallocate(std::size_t n) { _data.reallocate(n * BlockSize); }

  auto clear() { _data.clear(); }

  auto begin() const -> const_iterator {
    return iter::make_blocked_iterator<BlockSize>(_data.begin());
  }

  auto begin() -> iterator {
    return iter::make_blocked_iterator<BlockSize>(_data.begin());
  }

  auto end() const -> const_iterator {
    return iter::make_blocked_iterator<BlockSize>(_data.end());
  }

  auto end() -> iterator {
    return iter::make_blocked_iterator<BlockSize>(_data.end());
  }

  auto size() const -> size_type { return _data.size() / BlockSize; }

  auto empty() const -> bool { return size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(end() - 1); }

  auto back() -> reference { return *(end() - 1); }

  auto operator[](std::size_t i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](std::size_t i) -> reference { return *(begin() + i); }

  auto data_buffer() const -> const tf::buffer<T> & { return _data; }
  auto data_buffer() -> tf::buffer<T> & { return _data; }

private:
  tf::buffer<T> _data;
};

/// @ingroup core_buffers
/// @brief Runtime-block-size specialization.
///
/// The block size is fixed at construction; storage stays one contiguous
/// allocation, iteration yields @ref tf::range views over each block.
template <typename T> class blocked_buffer<T, tf::dynamic_size> {
public:
  using iterator = iter::blocked_iterator<T *, tf::dynamic_size>;
  using const_iterator = iter::blocked_iterator<const T *, tf::dynamic_size>;
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using size_type = typename std::iterator_traits<iterator>::difference_type;

  blocked_buffer() = default;
  explicit blocked_buffer(std::size_t block_size) : _block_size{block_size} {}
  blocked_buffer(tf::buffer<T> &&data, std::size_t block_size)
      : _data{std::move(data)}, _block_size{block_size} {}
  blocked_buffer(const tf::buffer<T> &data, std::size_t block_size)
      : _data{data}, _block_size{block_size} {}

  template <typename Iterator, std::size_t N>
  auto push_back(const tf::range<Iterator, N> &r) {
    for (std::size_t i = 0; i < _block_size; ++i)
      _data.push_back(r[i]);
  }

  auto reserve(std::size_t n) { _data.reserve(n * _block_size); }

  auto allocate(std::size_t n) { _data.allocate(n * _block_size); }

  auto reallocate(std::size_t n) { _data.reallocate(n * _block_size); }

  auto clear() { _data.clear(); }

  auto begin() const -> const_iterator {
    return iter::make_blocked_iterator(_data.begin(), _block_size);
  }

  auto begin() -> iterator {
    return iter::make_blocked_iterator(_data.begin(), _block_size);
  }

  auto end() const -> const_iterator {
    return iter::make_blocked_iterator(_data.end(), _block_size);
  }

  auto end() -> iterator {
    return iter::make_blocked_iterator(_data.end(), _block_size);
  }

  auto size() const -> size_type {
    return _block_size ? _data.size() / _block_size : 0;
  }

  auto empty() const -> bool { return size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(end() - 1); }

  auto back() -> reference { return *(end() - 1); }

  auto operator[](std::size_t i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](std::size_t i) -> reference { return *(begin() + i); }

  auto block_size() const -> std::size_t { return _block_size; }

  auto data_buffer() const -> const tf::buffer<T> & { return _data; }
  auto data_buffer() -> tf::buffer<T> & { return _data; }

private:
  tf::buffer<T> _data;
  std::size_t _block_size = 0;
};

} // namespace tf
