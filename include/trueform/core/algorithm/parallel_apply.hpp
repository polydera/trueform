/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "tbb/parallel_for.h"

namespace tf {
template <typename Iterator, typename Func>
auto parallel_apply(Iterator first, Iterator last, Func &&f) -> void {
  tbb::parallel_for(
      tbb::blocked_range<Iterator>(first, last),
      [f = static_cast<Func &&>(f)](const tbb::blocked_range<Iterator> &range) {
        for (Iterator it = range.begin(); it != range.end(); ++it) {
          if constexpr (std::is_integral<Iterator>::value)
            f(it);
          else
            f(*it);
        }
      });
}
/// @ingroup algorithms
/// @brief Applies a function to each element of a range in parallel.
///
/// This function applies the given functor `f` to each element of the input range `r`,
/// using parallel execution. Internally, it leverages @ref parallel_for to split
/// the range and apply the function concurrently.
///
/// @tparam Range A range type that supports random-access iteration.
/// @tparam Func A callable that takes a reference to an element of the range.
/// @param r The range to iterate over.
/// @param f The function to apply to each element in the range.
///
/// @note This is a parallel equivalent of a simple for-loop like:
/// @code
/// for (auto &elem : r) f(elem);
/// @endcode
///
/// @see @ref parallel_for
template <typename Range, typename Func>
auto parallel_apply(Range &&r, Func &&f) -> void {
  return parallel_apply(r.begin(), r.end(), static_cast<Func &&>(f));
}

template <typename Iterator0, typename Iterator1, typename Func>
auto parallel_apply(std::pair<Iterator0, Iterator1> first,
                    std::pair<Iterator0, Iterator1> last, Func &&f) -> void {
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, last - first),
                    [f = static_cast<Func &&>(f),
                     first](const tbb::blocked_range<std::size_t> &range) {
                      auto it_first = first.first + range.begin();
                      auto it_second = first.second + range.begin();
                      for (std::size_t i = range.begin(); i < range.end();
                           ++i) {
                        f(*it_first++, *it_second++);
                      }
                    });
}

template <typename Range0, typename Range1, typename Func>
auto parallel_apply(std::tuple<Range0, Range1> r, Func &&f) -> void {
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, std::get<0>(r).size()),
                    [f = static_cast<Func &&>(f),
                     &r](const tbb::blocked_range<std::size_t> &range) {
                      auto it_first = std::get<0>(r).begin() + range.begin();
                      auto it_second = std::get<1>(r).begin() + range.begin();
                      for (std::size_t i = range.begin(); i < range.end(); ++i)
                        f(*it_first++, *it_second++);
                    });
}
} // namespace tf
