/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
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
