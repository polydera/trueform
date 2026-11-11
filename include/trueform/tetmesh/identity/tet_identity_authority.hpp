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
#include "../../arrangement/identity/assert_welds_are_lawful.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/elect_identity_components.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <tuple>
#include <utility>

namespace tf::tetmesh {

/// THE DRAIN'S IDENTITY AUTHORITY — the one producer of a drain's point
/// identities: every identity is named, minted, seated or retired only
/// through its verbs, and every reader asks it. It is the arrangement's
/// identity substrate (@ref tf::arrangement::identity_substrate) on the
/// drain's one appending tier, its plane half having no carrier here.
///
/// One flat space: the entrance's originals `[0, n_originals)`, then the
/// created identities in the order they are minted, each at its lattice
/// position, with the pair of input faces it was minted for (`{-1, -1}` for
/// an original) and the kernel site it is seated at. THE EXACT LATTICE IS
/// THE IDENTITY: a statement names a point by its exact lattice position,
/// one identity per position, and only a position nothing names mints; the
/// reach grid is the entrance's (@ref state_tet_reach_cells) and the wave's
/// closure's, never a mint's. Topological identity and coordinate equality
/// are different facts: a point keeps the position it was stated at, and
/// its survivor speaks it. THE ELECTION
/// decides every coincidence by the substrate's keys
/// (@ref tf::intersect::graph::identity_key) — a standing created identity
/// first, then the originals, the smallest key of each — and a retirement
/// relabels the loser into the winner, which keeps its position. A death is
/// stated nowhere: an identity no published cell names is dead by
/// definition (@ref account_tet_identities).
///
/// The retirements are a dense parent lane, chased to the standing
/// identity; @ref close ends a statement, compressing every chase it made
/// to one step and stating the closed rows `{1, id, target}` sorted, the
/// substrate's shape (@ref rows). The positions are one sorted table,
/// extended once per statement.
template <typename Index, typename Int> class tet_identity_authority {
public:
  using pt3_t = tf::exact::pt3<Int>;
  using pair_t = std::array<Index, 2>;
  using row_t = std::array<Index, 3>;
  static constexpr Index none = Index(-1);

  // ---- the reads -----------------------------------------------------

  auto n_identities() const -> Index { return Index(_parent.size()); }
  auto n_originals() const -> Index { return _n_originals; }
  auto n_created() const -> Index { return n_identities() - _n_originals; }
  auto is_original(Index id) const -> bool { return id < _n_originals; }

  /// The identity `id` speaks now. O(the chase, one step after a close).
  auto standing(Index id) const -> Index {
    while (_parent[std::size_t(id)] != none)
      id = _parent[std::size_t(id)];
    return id;
  }
  /// The identity `id` was retired into, `-1` for a standing one.
  auto target(Index id) const -> Index {
    return _parent[std::size_t(id)] == none ? none : standing(id);
  }
  auto position_of(Index id) const -> const pt3_t & {
    return _positions[std::size_t(id)];
  }
  /// The pair of input faces `id` was minted for, `{-1, -1}` for an
  /// original.
  auto pair_of(Index id) const -> const pair_t & {
    return _pairs[std::size_t(id)];
  }
  /// The site the identity `id` speaks is seated at, `-1` for none.
  auto slot_of(Index id) const -> Index {
    return _slot[std::size_t(standing(id))];
  }
  /// The site `id` itself was seated at, retired or not, `-1` for none.
  auto seat_of(Index id) const -> Index { return _slot[std::size_t(id)]; }
  /// The identity seated at site `slot`, `-1` for none. O(1).
  auto identity_at(Index slot) const -> Index {
    return std::size_t(slot) < _at_slot.size() ? _at_slot[std::size_t(slot)]
                                               : none;
  }
  /// Whether the site `slot` holds a standing created identity minted for
  /// `pair`: the reach election's filter, the walk being the kernel's
  /// (@ref tf::topology::cdt::dt3::find_tet_site_within). O(1).
  auto minted_for(Index slot, const pair_t &pair) const -> bool {
    const Index id = identity_at(slot);
    return id != none && !is_original(id) && _parent[std::size_t(id)] == none &&
           _pairs[std::size_t(id)] == pair;
  }
  /// The identity standing at exactly `p`, `-1` where none is named there:
  /// one binary search.
  auto standing_at(const pt3_t &p) const -> Index {
    const Index named = named_at(p);
    return named == none ? none : standing(named);
  }
  /// THE ELECTION of two standing identities: the one the smaller key
  /// speaks, a created identity before an original. O(1).
  auto elect(Index a, Index b) const -> Index {
    return key_of(a) < key_of(b) ? a : b;
  }
  /// The closed rows of the last @ref close, `{1, id, target}` by id.
  auto rows() const -> const tf::buffer<row_t> & { return _rows; }

  // ---- the verbs -----------------------------------------------------

  /// The entrance's points as originals, original `i` at `points[i]` and
  /// seated at `slots[i]`, one original per position as the entrance states
  /// them (@ref state_tet_reach_cells). Once, before any other verb: the
  /// lanes and the seats in parallel over the originals and one parallel
  /// sort. O(n log n).
  template <typename Points, typename Slots>
  auto name_originals(const Points &points, const Slots &slots) -> void {
    if constexpr (tf::core::verification)
      tf::core::verify(_parent.size() == 0, "the originals are named first");
    const std::size_t n = points.size();
    _n_originals = Index(n);
    _positions.allocate(n);
    _pairs.allocate(n);
    _parent.allocate(n);
    _slot.allocate(n);
    _table.allocate(n);
    tf::parallel_for_each(
        tf::make_sequence_range(n),
        [this, &points, &slots](std::size_t i) {
          _positions[i] = points[i];
          _pairs[i] = {none, none};
          _parent[i] = none;
          _slot[i] = Index(slots[i]);
          _table[i] = {points[i], Index(i)};
        },
        tf::checked);
    tbb::parallel_sort(_table.begin(), _table.end());
    if constexpr (tf::core::verification)
      for (std::size_t t = 1; t < n; ++t)
        tf::core::verify(_table[t].position != _table[t - 1].position,
                         "the entrance states one original per position");
    const Index sites =
        n ? tf::reduce(
                _slot, [](Index a, Index b) { return std::max(a, b); },
                Index(-1), tf::checked) +
                1
          : Index(0);
    _at_slot.allocate(std::size_t(sites));
    tf::parallel_fill(_at_slot, none);
    tf::parallel_for_each(
        tf::make_sequence_range(n),
        [this](std::size_t id) {
          _at_slot[std::size_t(_slot[id])] = Index(id);
        },
        tf::checked);
    close();
  }

  /// A statement's points by exact position: per entry the identity it
  /// names into `ids` — the one named at its position, or one minted there,
  /// one per position, minted in the order of each position's first entry,
  /// for the smallest pair its entries state. Returns the first minted. O(m log
  /// m) in the statement's entries, plus one extension of the table.
  template <typename Points, typename Pairs>
  auto name_positions(const Points &points, const Pairs &pairs,
                      tf::buffer<Index> &ids) -> Index {
    const std::size_t m = points.size();
    const Index first = n_identities();
    ids.allocate(m);
    auto &fresh = _fresh;
    fresh.allocate(m);
    tf::parallel_for_each(
        tf::make_sequence_range(m),
        [&fresh, &points](std::size_t i) {
          fresh[i] = {pt3_t(points[i]), Index(i)};
        },
        tf::checked);
    tbb::parallel_sort(fresh.begin(), fresh.end());
    auto &births = _births;
    births.clear();
    for (std::size_t i = 0; i < m;) {
      std::size_t j = i + 1;
      while (j < m && fresh[j].position == fresh[i].position)
        ++j;
      const Index named = named_at(fresh[i].position);
      if (named != none)
        for (std::size_t k = i; k < j; ++k)
          ids[std::size_t(fresh[k].id)] = named;
      else
        births.push_back({fresh[i].id, Index(i), Index(j)});
      i = j;
    }
    std::sort(births.begin(), births.end());
    auto &grown = _grown;
    grown.clear();
    for (const auto &birth : births) {
      const Index id = n_identities();
      pair_t pair = pairs[std::size_t(birth[0])];
      for (Index k = birth[1]; k < birth[2]; ++k) {
        const Index entry = fresh[std::size_t(k)].id;
        pair = std::min(pair, pair_t(pairs[std::size_t(entry)]));
        ids[std::size_t(entry)] = id;
      }
      const auto &head = fresh[std::size_t(birth[1])];
      _positions.push_back(head.position);
      _pairs.push_back(pair);
      _parent.push_back(none);
      _slot.push_back(none);
      grown.push_back({head.position, id});
    }
    std::sort(grown.begin(), grown.end());
    auto &merged = _fresh;
    merged.allocate(_table.size() + grown.size());
    std::merge(_table.begin(), _table.end(), grown.begin(), grown.end(),
               merged.begin());
    std::swap(_table, merged);
    return first;
  }

  /// The identity `id` seated at site `slot`. O(1).
  auto seat(Index id, Index slot) -> void {
    _slot[std::size_t(id)] = slot;
    if (std::size_t(slot) >= _at_slot.size()) {
      const std::size_t known = _at_slot.size();
      _at_slot.reallocate(std::size_t(slot) + 1);
      std::fill(_at_slot.begin() + std::ptrdiff_t(known), _at_slot.end(), none);
    }
    _at_slot[std::size_t(slot)] = id;
  }

  /// The standing identity `id` retired into the standing identity `into`,
  /// the survivor the caller's law names — the election's winner where two
  /// identities are welded (@ref weld), the original corner where the
  /// census convicts a created identity in its cell: every name of
  /// `id` speaks `into` from now on, at `into`'s position and site. O(1).
  auto retire(Index id, Index into) -> void {
    if constexpr (tf::core::verification)
      tf::core::verify(id != into && _parent[std::size_t(id)] == none &&
                           _parent[std::size_t(into)] == none,
                       "a retirement relabels a standing identity into "
                       "another standing one");
    _parent[std::size_t(id)] = into;
    _retired.push_back(id);
  }
  /// Two standing identities stated one: the loser retired into the
  /// election's winner, which is returned. O(1).
  auto weld(Index a, Index b) -> Index {
    a = standing(a);
    b = standing(b);
    if (a == b)
      return a;
    const Index winner = elect(a, b);
    retire(winner == a ? b : a, winner);
    return winner;
  }

  /// The end of a statement: every retirement chased to one step, and the
  /// closed rows stated, sorted by id. The statement's own retirements are
  /// sorted, the standing rows retargeted where their target retired, and
  /// the two merged: O(r log r + R), r the statement's retirements and R the
  /// rows standing.
  auto close() -> void {
    auto &stated = _retired;
    std::sort(stated.begin(), stated.end());
    for (const Index id : stated)
      _parent[std::size_t(id)] = standing(id);
    tf::parallel_for_each(
        _rows,
        [this](row_t &row) {
          const Index above = _parent[std::size_t(row[2])];
          if (above != none) {
            row[2] = above;
            _parent[std::size_t(row[1])] = above;
          }
        },
        tf::checked);
    auto &fresh = _stated_rows;
    fresh.allocate(stated.size());
    for (std::size_t i = 0; i < stated.size(); ++i)
      fresh[i] = {Index(1), stated[i], _parent[std::size_t(stated[i])]};
    auto &merged = _merged_rows;
    merged.allocate(_rows.size() + fresh.size());
    std::merge(_rows.begin(), _rows.end(), fresh.begin(), fresh.end(),
               merged.begin(),
               [](const row_t &a, const row_t &b) { return a[1] < b[1]; });
    std::swap(_rows, merged);
    stated.clear();
    if constexpr (tf::core::verification)
      tf::arrangement::assert_welds_are_lawful(
          _rows, n_identities(),
          [this](Index id) { return _parent[std::size_t(id)] == none; });
  }

private:
  struct position_entry {
    pt3_t position;
    Index id;
    friend auto operator<(const position_entry &x, const position_entry &y)
        -> bool {
      return std::tie(x.position[0], x.position[1], x.position[2], x.id) <
             std::tie(y.position[0], y.position[1], y.position[2], y.id);
    }
  };
  auto named_at(const pt3_t &p) const -> Index {
    const auto at = std::lower_bound(_table.begin(), _table.end(),
                                     position_entry{p, Index(-1)});
    return at != _table.end() && at->position == p ? at->id : none;
  }
  auto key_of(Index id) const -> tf::intersect::graph::identity_key<Index> {
    return is_original(id) ? tf::intersect::graph::original_identity_key(id)
                           : tf::intersect::graph::standing_identity_key(id);
  }

  Index _n_originals = 0;
  tf::buffer<pt3_t> _positions;
  tf::buffer<pair_t> _pairs;
  tf::buffer<Index> _parent;
  tf::buffer<Index> _slot;
  tf::buffer<Index> _at_slot;
  tf::buffer<position_entry> _table;
  tf::buffer<Index> _retired;
  tf::buffer<row_t> _rows;
  tf::buffer<row_t> _stated_rows;
  tf::buffer<row_t> _merged_rows;
  tf::buffer<position_entry> _fresh;
  tf::buffer<position_entry> _grown;
  tf::buffer<std::array<Index, 3>> _births;
};

} // namespace tf::tetmesh
