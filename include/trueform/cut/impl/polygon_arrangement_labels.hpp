/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
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
