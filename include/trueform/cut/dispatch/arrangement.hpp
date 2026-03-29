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
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/zip.hpp"
#include "./boolean.hpp"
#include "tbb/task_group.h"

namespace tf::cut::dispatch {

template <typename Range, typename F>
auto arrangement(const Range &forms, F &&f) {
  using S = decltype(make_missing_structures(forms[0]));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return f(forms);
  } else {
    auto n = forms.size();
    tf::small_vector<S, 10> structs(n);

    tbb::task_group tg;
    for (std::size_t i = 0; i < n; ++i)
      tg.run([&, i] { structs[i] = make_missing_structures(forms[i]); });
    tg.wait();

    return f(tf::make_mapped_range(
        tf::zip(forms, structs), [](auto pair) {
          auto &&[form, s] = pair;
          return tag_with_structures(form, s);
        }));
  }
}

} // namespace tf::cut::dispatch
