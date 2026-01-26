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
#include "./mapped_iterator.hpp"
namespace tf::iter {
template <typename Iterator, typename RandomIterator> struct indirect_policy {
  auto
  operator()(typename std::iterator_traits<Iterator>::reference index) const
      -> decltype(auto) {
    using diff_t = typename std::iterator_traits<RandomIterator>::difference_type;
    return data_iter[static_cast<diff_t>(index)];
  }
  RandomIterator data_iter;
};

template <typename Iterator, typename RandomIterator>
using indirect = mapped<Iterator, indirect_policy<Iterator, RandomIterator>>;

template <typename Iterator, typename RandomIterator>
auto make_indirect(Iterator iter, RandomIterator data_iter) {
  return make_mapped(std::move(iter), indirect_policy<Iterator, RandomIterator>{
                                          std::move(data_iter)});
}
} // namespace tf::iter
