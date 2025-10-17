/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/buffer.hpp"

namespace tf::cut {
template <typename LabelType> struct polygon_arrangement_labels {
  tf::buffer<LabelType> polygon_labels;
  tf::buffer<LabelType> cut_labels;
  LabelType n_components;
};
} // namespace tf::cut
