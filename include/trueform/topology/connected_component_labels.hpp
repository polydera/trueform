/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/buffer.hpp"
namespace tf {
template <typename LabelType> struct connected_component_labels {
  tf::buffer<LabelType> labels;
  LabelType n_components;
};
} // namespace tf
