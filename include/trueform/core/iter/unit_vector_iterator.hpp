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
#include "../unit_vector.hpp"
#include "../unit_vector_view.hpp"
#include "./stride_iterator_api.hpp"

namespace tf::iter {
template <typename Iterator, std::size_t BlockSize>
struct unit_vector_api_handle {
public:
  using iterator_category =
      typename std::iterator_traits<Iterator>::iterator_category;
  using element_t = typename std::iterator_traits<Iterator>::value_type;
  using element_view_t = std::remove_reference_t<
      typename std::iterator_traits<Iterator>::reference>;
  using reference = tf::unit_vector_view<element_view_t, BlockSize>;
  using value_type = tf::unit_vector<element_t, BlockSize>;
  using pointer = void;
  using difference_type =
      typename std::iterator_traits<Iterator>::difference_type;

  unit_vector_api_handle() = default;
  unit_vector_api_handle(Iterator iter) : iter{std::move(iter)} {}
  auto base_iter() const -> const Iterator & { return iter; }
  auto base_iter() -> Iterator & { return iter; }
  constexpr auto iterator_stride() const -> std::size_t { return BlockSize; }
  auto dereference() const -> reference {
    return tf::make_unit_vector_view<BlockSize>(
        tf::unsafe, tf::make_vector_view<BlockSize>(&(*iter)));
  }

private:
  Iterator iter;
};

template <typename Iterator, std::size_t BlockSize>
struct unit_vector_iterator
    : stride_api<unit_vector_iterator<Iterator, BlockSize>,
                 unit_vector_api_handle<Iterator, BlockSize>> {
private:
  using base_t = stride_api<unit_vector_iterator<Iterator, BlockSize>,
                            unit_vector_api_handle<Iterator, BlockSize>>;
  using handle_t = unit_vector_api_handle<Iterator, BlockSize>;

public:
  explicit unit_vector_iterator(Iterator iter)
      : base_t{handle_t{std::move(iter)}} {}
  unit_vector_iterator() = default;
};

template <std::size_t BlockSize, typename Iterator>
auto make_unit_vector_iterator(Iterator iter) {
  return unit_vector_iterator<Iterator, BlockSize>{std::move(iter)};
}

} // namespace tf::iter
