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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/checked.hpp"
#include "./nifti_header.hpp"
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace tf::io::nifti {

template <typename Src, typename Dst, bool Swap, bool Scale>
auto convert_block(const char *src, Dst *dst, std::size_t n, float slope,
                   float intercept) -> void {
  tf::parallel_for(
      std::size_t{0}, n,
      [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
          Src value;
          std::memcpy(&value, src + i * sizeof(Src), sizeof(Src));
          if constexpr (Swap)
            value = swapped_bytes(value);
          if constexpr (Scale)
            dst[i] = static_cast<Dst>(double(slope) * double(value) +
                                      double(intercept));
          else
            dst[i] = static_cast<Dst>(value);
        }
      },
      tf::checked);
}

/// One pass from the file's samples to the volume's, the byte swap and the
/// scl scaling folded into it; the runtime flags select a specialization so
/// the average path's inner loop is a bare copy.
template <typename Src, typename Dst>
auto convert_samples(const char *src, Dst *dst, std::size_t n, bool swap,
                     bool scale, float slope, float intercept) -> void {
  if constexpr (std::is_same_v<Src, Dst>) {
    if (!swap && !scale) {
      // Chunked so fresh destination pages fault in parallel.
      tf::parallel_for(
          std::size_t{0}, n,
          [&](std::size_t begin, std::size_t end) {
            std::memcpy(dst + begin, src + begin * sizeof(Dst),
                        (end - begin) * sizeof(Dst));
          },
          tf::checked);
      return;
    }
  }
  if (swap) {
    if (scale)
      convert_block<Src, Dst, true, true>(src, dst, n, slope, intercept);
    else
      convert_block<Src, Dst, true, false>(src, dst, n, slope, intercept);
  } else {
    if (scale)
      convert_block<Src, Dst, false, true>(src, dst, n, slope, intercept);
    else
      convert_block<Src, Dst, false, false>(src, dst, n, slope, intercept);
  }
}

} // namespace tf::io::nifti
