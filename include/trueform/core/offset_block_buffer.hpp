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
#include "./views/offset_block_range.hpp"

namespace tf {

/// @ingroup core_buffers
/// @brief A buffer of variable-sized blocks using offset-based storage.
///
/// Stores data contiguously with a separate offset array indicating
/// where each block begins. Iteration yields @ref tf::range views
/// over each variable-length block.
///
/// @tparam Index The offset index type.
/// @tparam T The data element type.
template <typename Index, typename T> class offset_block_buffer {
public:
  using iterator = decltype(views::make_offset_block_begin(
      std::declval<tf::buffer<Index> &>(), std::declval<tf::buffer<T> &>()));
  using const_iterator = decltype(views::make_offset_block_begin(
      std::declval<const tf::buffer<Index> &>(),
      std::declval<const tf::buffer<T> &>()));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using size_type = std::size_t;

  template <typename Iterator, std::size_t Size>
  auto push_back(const tf::range<Iterator, Size> &r) {
    if (!_offsets.size())
      _offsets.push_back(0);
    auto old_size = _data.size();
    _data.reallocate(old_size + r.size());
    std::copy(r.begin(), r.end(), _data.begin() + old_size);
    _offsets.push_back(_data.size());
  }

  template <typename U> auto push_back(const std::initializer_list<U> &r) {
    if (!_offsets.size())
      _offsets.push_back(0);
    auto old_size = _data.size();
    _data.reallocate(old_size + r.size());
    std::copy(r.begin(), r.end(), _data.begin() + old_size);
    _offsets.push_back(_data.size());
  }

  auto begin() const -> const_iterator {
    return views::make_offset_block_begin(_offsets, _data);
  }

  auto begin() -> iterator {
    return views::make_offset_block_begin(_offsets, _data);
  }

  auto end() const -> const_iterator {
    return views::make_offset_block_end(_offsets, _data);
  }

  auto end() -> iterator {
    return views::make_offset_block_end(_offsets, _data);
  }

  auto size() const -> size_type {
    return _offsets.size() - (_offsets.size() != 0);
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

  auto offsets_buffer() const -> const tf::buffer<Index> & { return _offsets; }
  auto offsets_buffer() -> tf::buffer<Index> & { return _offsets; }
  auto data_buffer() const -> const tf::buffer<T> & { return _data; }
  auto data_buffer() -> tf::buffer<T> & { return _data; }

  auto clear() {
    _offsets.clear();
    _data.clear();
  }

private:
  tf::buffer<Index> _offsets;
  tf::buffer<T> _data;
};

} // namespace tf
