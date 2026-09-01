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
#include "../buffer.hpp"
#include <cstddef>

namespace tf {

/// @ingroup core_algorithms
/// @brief A parallel block's accumulation over a dense global axis it touches
///        sparsely.
///
/// A block-local dense array over the axis makes every block pay the WHOLE
/// axis twice — once to clear it, once to fold it back — to carry the handful
/// of keys it actually names. This keeps only those keys, as runs: the labels
/// a block walks are spatially coherent, so consecutive elements land on the
/// same entry, and the fold costs what the block touched.
///
/// `touch` scans back at most `scan_window` entries, so a block naming no more
/// than that many keys accumulates each of them EXACTLY ONCE and an
/// accumulator that is not associative keeps its per-block grouping; beyond
/// the window a key's entries split and the fold sums the parts. The returned
/// reference is valid until the next `touch`.
///
/// This is the block-local answer. Where one dense table shared by the whole
/// pass is the right call instead — cleared once and reused across groups by a
/// watermark rather than per group — see the `point_sentinel` walk in
/// @ref tf::make_csg_domains.
template <typename Index, typename T> struct sparse_block_accumulator {
  struct entry {
    Index key;
    T value;
  };
  static constexpr std::size_t scan_window = 32;

  tf::buffer<entry> entries;

  auto clear() -> void { entries.clear(); }

  auto touch(Index key, const T &identity) -> T & {
    const auto n = entries.size();
    const auto scan = n < scan_window ? n : scan_window;
    for (std::size_t k = 1; k <= scan; ++k)
      if (entries[n - k].key == key)
        return entries[n - k].value;
    entries.push_back({key, identity});
    return entries[entries.size() - 1].value;
  }
};

} // namespace tf
