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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/constant.hpp"

namespace tf::topology {
template <typename Index, typename LabelType = short>
class sequential_connected_components_finder {
  using label_t = LabelType;

public:
  template <typename Range0, typename Range1, typename F>
  auto run(Range0 &&labels, const Range1 &mask, const F &applier) -> label_t {
    tf::parallel_fill(labels, -1);
    label_t label = 0;
    for (Index i = 0; i < Index(labels.size()); ++i) {
      if (labels[i] != -1 || !mask[i])
        continue;
      _stack.push_back(i);
      while (_stack.size()) {
        auto next = _stack.back();
        _stack.pop_back();
        if (labels[next] != -1)
          continue;
        labels[next] = label;
        applier(next, [&](Index id) {
          if (mask[id])
            _stack.push_back(id);
        });
      }
      label++;
    }
    return label;
  }

  template <typename Range, typename F>
  auto run(Range &&labels, const F &applier) -> Index {
    return run(labels, tf::make_constant_range(true, labels.size()), applier);
  }

private:
  tf::buffer<Index> _stack;
};
} // namespace tf::topology
