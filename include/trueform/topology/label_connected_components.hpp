/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./components/finder.hpp"
#include "./components/sequential_finder.hpp"
#include "./connected_component_labels.hpp"

namespace tf {
template <typename Index, typename Range0, typename Range1, typename F>
auto label_connected_components_masked(Range0 &&labels, const Range1 &mask,
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

template <typename Index, typename LabelType, typename Range1, typename F>
auto label_connected_components_masked(tf::connected_component_labels<LabelType> &cl,
                                const Range1 &mask, const F &applier,
                                Index expected_number_of_components = 2) {
  cl.n_components = label_connected_components_masked(
      cl.labels, mask, applier, expected_number_of_components);
}

template <typename Index, typename Range, typename F>
auto label_connected_components(Range &&labels, const F &applier,
                                Index expected_number_of_components = 2) {
  return label_connected_components_masked(
      labels, tf::make_constant_range(true, labels.size()), applier,
      expected_number_of_components);
}

template <typename Index, typename LabelType, typename F>
auto label_connected_components(tf::connected_component_labels<LabelType> &cl,
                                const F &applier,
                                Index expected_number_of_components = 2) {
  cl.n_components = label_connected_components_masked(cl.labels, applier,
                                               expected_number_of_components);
}
} // namespace tf
