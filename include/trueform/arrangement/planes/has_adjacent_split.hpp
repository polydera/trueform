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

#include <algorithm>
#include <type_traits>

namespace tf::arrangement {

/// @brief Whether the edge `key` already carries a split at `parameter`.
///
/// Two splits are the same split when they land on the same point of the
/// grid the transported parameter defines, or on neighbouring ones. The
/// tolerance is one quantum of that grid, so it follows the width by
/// construction rather than through a second constant of its own.
///
/// `key_of` and `param_of` read the edge and the position off a split, so
/// the question is asked in whatever currency the caller's table states its
/// edges in; `sorted` is ordered by that same key.
///
/// The question is asked while splits are being produced, so it spans two
/// tables: `sorted`, which answers by binary search, and `fresh`, the ones
/// this wave has appended and nobody has ordered yet. Searching an appended
/// table as if it were ordered is unsound in both directions — it can return
/// a span of unrelated edges, admitting a duplicate onto one edge, or
/// suppress a real split against one that is not on this edge at all.
template <typename Key, typename Param, typename Splits, typename KeyOf,
          typename ParamOf>
auto has_adjacent_split(const Key &key, Param parameter, const Splits &sorted,
                        const Splits &fresh, const KeyOf &key_of,
                        const ParamOf &param_of) -> bool {
  using split_t = std::decay_t<decltype(*sorted.begin())>;
  struct compare {
    const KeyOf &key_of;
    auto operator()(const split_t &split, const Key &candidate) const -> bool {
      return key_of(split) < candidate;
    }
    auto operator()(const Key &candidate, const split_t &split) const -> bool {
      return candidate < key_of(split);
    }
  };
  auto adjacent = [&](const split_t &split) {
    const Param distance = Param(param_of(split)) - parameter;
    return distance <= Param(1) && distance >= Param(-1);
  };
  const auto range =
      std::equal_range(sorted.begin(), sorted.end(), key, compare{key_of});
  for (auto split = range.first; split != range.second; ++split)
    if (adjacent(*split))
      return true;
  for (const auto &split : fresh)
    if (key_of(split) == key && adjacent(split))
      return true;
  return false;
}

/// @brief Whether the edge `key` already carries a split at `parameter`,
///        over one ordered table.
template <typename Key, typename Param, typename Splits, typename KeyOf,
          typename ParamOf>
auto has_adjacent_split(const Key &key, Param parameter, const Splits &sorted,
                        const KeyOf &key_of, const ParamOf &param_of) -> bool {
  return has_adjacent_split(key, parameter, sorted, Splits{}, key_of, param_of);
}

} // namespace tf::arrangement
