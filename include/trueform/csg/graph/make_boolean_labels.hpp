/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include <cstdint>

namespace tf::csg::graph {

/// @brief The boolean's per-face operand label: `0` for the left
///        operand, `1` for the right. Narrowed from the arrangement's
///        tag labels, which carry the same fact in the index type.
template <typename Tags>
auto make_boolean_labels(const Tags &tag_labels) -> tf::buffer<std::int8_t> {
  tf::buffer<std::int8_t> labels;
  labels.allocate(tag_labels.size());
  tf::parallel_transform(tag_labels, labels, [](auto tag) {
    return static_cast<std::int8_t>(tag);
  });
  return labels;
}

} // namespace tf::csg::graph
