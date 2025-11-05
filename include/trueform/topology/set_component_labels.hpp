/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./connected_component_labels.hpp"
#include "./set_type.hpp"

namespace tf {
template <typename LabelType> struct set_component_labels {
  tf::connected_component_labels<LabelType> component_labels;
  tf::buffer<tf::set_type> set_types;
};
} // namespace tf
