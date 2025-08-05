/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./components/finder.hpp"
#include "./components/sequential_finder.hpp"

namespace tf {
template <typename Index, typename Range0, typename Range1, typename F>
auto label_connected_components(Range0 &&labels, const Range1 &mask,
                                const F &applier,
                                Index expected_number_of_components = 2) {
  using label_t = std::decay_t<decltype(labels[0])>;
  if (labels.size() < 5000 || expected_number_of_components > 200) {
    tf::topology::sequential_connected_components_finder<Index, label_t> finder;
    return finder.run(labels, mask, applier);
  } else {
    tf::topology::connected_components_finder<Index, label_t> finder;
    return finder.run(labels, mask, applier);
  }
}

template <typename Index, typename Range, typename F>
auto label_connected_components(Range &&labels, const F &applier,
                                Index expected_number_of_components = 2) {
  return label_connected_components(
      labels, tf::make_constant_range(true, labels.size()), applier,
      expected_number_of_components);
}
} // namespace tf
