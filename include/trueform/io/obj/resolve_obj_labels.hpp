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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tf::io::obj {

template <typename Index>
auto resolve_obj_labels(tf::buffer<Index> &source,
                        std::vector<std::string> &names, bool needs_default,
                        tf::buffer<Index> &output,
                        std::vector<std::string> &output_names) -> void {
  if (names.empty())
    return;
  if (!needs_default) {
    output = std::move(source);
    output_names = std::move(names);
    return;
  }

  output_names.reserve(names.size() + 1);
  output_names.emplace_back("default");
  for (auto &name : names)
    output_names.push_back(std::move(name));
  output.allocate(source.size());
  tf::parallel_for_each(
      tf::enumerate(source),
      [&](auto pair) {
        auto &&[index, label] = pair;
        output[index] = label == Index(-1) ? Index(0) : Index(label + 1);
      },
      tf::checked);
}

} // namespace tf::io::obj
