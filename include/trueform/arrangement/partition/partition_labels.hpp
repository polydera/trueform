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
#include "../../core/buffer.hpp"

namespace tf::arrangement {

template <typename LabelType> struct partition_labels {
  tf::buffer<LabelType> polygon_labels;
  tf::buffer<LabelType> cut_labels;
  LabelType n_components;
};

} // namespace tf::arrangement
