/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./mapped_iterator.hpp"
namespace tf::iter {
template <typename Iterator, typename RandomIterator> struct indirect_policy {
  auto
  operator()(typename std::iterator_traits<Iterator>::reference index) const
      -> decltype(auto) {
    return data_iter[index];
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
