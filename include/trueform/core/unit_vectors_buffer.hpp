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
#include "./iter/unit_vector_iterator.hpp"
#include "./unit_vectors.hpp"
namespace tf {

/// @ingroup core_buffers
/// @brief An owning buffer of unit vectors.
///
/// Stores unit vectors as interleaved coordinates and provides iteration
/// as @ref tf::unit_vector_view elements. Use `unit_vectors()` to obtain a
/// @ref tf::unit_vectors range.
///
/// @tparam T The coordinate scalar type.
/// @tparam Dims The number of dimensions per unit vector.
template <typename T, std::size_t Dims> class unit_vectors_buffer {
public:
  using iterator = tf::iter::unit_vector_iterator<T *, Dims>;
  using const_iterator = tf::iter::unit_vector_iterator<const T *, Dims>;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using value_type = tf::unit_vector<T, Dims>;
  using size_type = std::size_t;

  unit_vectors_buffer() = default;
  unit_vectors_buffer(const tf::buffer<T> &_data) : _raw_buffer{_data} {}
  unit_vectors_buffer(tf::buffer<T> &&_data) : _raw_buffer{std::move(_data)} {}

  template <typename Policy>
  auto push_back(const tf::unit_vector_like<Dims, Policy> &pt) {
    for (std::size_t i = 0; i < Dims; ++i)
      _raw_buffer.push_back(pt[i]);
  }

  template <typename... Ts>
  auto emplace_back(Ts &&...ts)
      -> std::enable_if_t<(sizeof...(Ts) == Dims), void> {
    (_raw_buffer.push_back(ts), ...);
  }

  auto erase(iterator from, iterator to) {
    _raw_buffer.erase(from.base_iter(), to.base_iter());
  }

  auto reserve(std::size_t n) { _raw_buffer.reserve(n * Dims); }

  auto allocate(std::size_t n) { _raw_buffer.allocate(n * Dims); }

  auto reallocate(std::size_t n) { _raw_buffer.reallocate(n * Dims); }

  auto clear() { _raw_buffer.clear(); }

  auto data_buffer() const -> const tf::buffer<T> & { return _raw_buffer; }

  auto data_buffer() -> tf::buffer<T> & { return _raw_buffer; }

  auto begin() const -> const_iterator {
    return tf::iter::make_unit_vector_iterator<Dims>(_raw_buffer.begin());
  }

  auto begin() -> iterator {
    return tf::iter::make_unit_vector_iterator<Dims>(_raw_buffer.begin());
  }

  auto end() const -> const_iterator {
    return tf::iter::make_unit_vector_iterator<Dims>(_raw_buffer.end());
  }

  auto end() -> iterator {
    return tf::iter::make_unit_vector_iterator<Dims>(_raw_buffer.end());
  }

  auto size() const -> size_type { return _raw_buffer.size() / Dims; }

  auto empty() const -> bool { return size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(end() - 1); }

  auto back() -> reference { return *(end() - 1); }

  auto operator[](std::size_t i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](std::size_t i) -> reference { return *(begin() + i); }

  auto unit_vectors() const { return tf::make_unit_vectors(*this); }

  auto unit_vectors() { return tf::make_unit_vectors(*this); }

private:
  tf::buffer<T> _raw_buffer;
};

} // namespace tf
