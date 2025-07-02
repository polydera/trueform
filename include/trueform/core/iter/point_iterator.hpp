/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../point.hpp"
#include "../point_view.hpp"
#include "./stride_iterator_api.hpp"

namespace tf::iter {
template <typename Iterator, std::size_t BlockSize> struct point_api_handle {
public:
  using iterator_category =
      typename std::iterator_traits<Iterator>::iterator_category;
  using element_t = typename std::iterator_traits<Iterator>::value_type;
  using reference = tf::point_view<element_t, BlockSize>;
  using value_type = tf::point<element_t, BlockSize>;
  using pointer = void;
  using difference_type =
      typename std::iterator_traits<Iterator>::difference_type;

  point_api_handle() = default;
  point_api_handle(Iterator iter) : iter{std::move(iter)} {}
  auto base_iter() const -> const Iterator & { return iter; }
  auto base_iter() -> Iterator & { return iter; }
  constexpr auto iterator_stride() const -> std::size_t { return BlockSize; }
  auto dereference() const -> reference {
    return tf::point_view<element_t, BlockSize>(&(*iter));
  }

private:
  Iterator iter;
};

template <typename Iterator, std::size_t BlockSize>
struct point_iterator : stride_api<point_iterator<Iterator, BlockSize>,
                                   point_api_handle<Iterator, BlockSize>> {
private:
  using base_t = stride_api<point_iterator<Iterator, BlockSize>,
                            point_api_handle<Iterator, BlockSize>>;
  using handle_t = point_api_handle<Iterator, BlockSize>;

public:
  explicit point_iterator(Iterator iter) : base_t{handle_t{std::move(iter)}} {}
};

template <std::size_t BlockSize, typename Iterator>
auto make_point_iterator(Iterator iter) {
  return point_iterator<Iterator, BlockSize>{std::move(iter)};
}

} // namespace tf::iter
