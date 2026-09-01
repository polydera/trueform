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

#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/slice.hpp"
#include "./plane_edge_def.hpp"
#include <cstddef>

namespace tf::intersect::graph {

/// The definition tables a plane tier owns: the definitions canon-major,
/// the canonical spans over them, and the plane CSR into them.
template <typename Index, typename Int> class plane_tables {
public:
  using def_t = plane_edge_def<Index>;

  auto edge_defs() const { return tf::make_range(_defs); }
  auto n_canon() const -> Index { return _n_canon; }
  auto canon_group(Index id) const {
    return tf::slice(tf::make_range(_defs),
                     std::size_t(_def_offsets[std::size_t(id)]),
                     std::size_t(_def_offsets[std::size_t(id) + 1]));
  }
  /// A plane's definitions, in canonical order.
  auto plane_edges(Index p) const {
    return tf::make_range(_edges)[std::size_t(p)];
  }
  /// The plane carrier space this tier answers in: its CSR's own extent.
  auto n_planes() const -> Index { return Index(_edges.size()); }

  auto defs() -> tf::buffer<def_t> & { return _defs; }
  auto defs() const -> const tf::buffer<def_t> & { return _defs; }
  auto def_offsets() -> tf::buffer<Index> & { return _def_offsets; }
  auto def_offsets() const -> const tf::buffer<Index> & { return _def_offsets; }
  auto edges() -> tf::offset_block_buffer<Index, Index> & { return _edges; }
  auto edges() const -> const tf::offset_block_buffer<Index, Index> & {
    return _edges;
  }
  auto n_canon() -> Index & { return _n_canon; }

  /// Whether both offset structures index exactly what they claim: the
  /// canonical spans over the definitions, and the plane CSR over its rows.
  /// A tier that holds nothing states it by holding no offsets at all.
  auto well_formed() const -> bool {
    const auto bounds = [](const tf::buffer<Index> &offsets, std::size_t n_data,
                           std::size_t n_blocks) {
      if (offsets.size() == 0)
        return n_blocks == 0 && n_data == 0;
      return offsets.size() == n_blocks + 1 && offsets[0] == Index(0) &&
             offsets[offsets.size() - 1] == Index(n_data);
    };
    if (_n_canon < Index(0))
      return false;
    return bounds(_def_offsets, _defs.size(), std::size_t(_n_canon)) &&
           bounds(_edges.offsets_buffer(), _edges.data_buffer().size(),
                  _edges.size());
  }

private:
  tf::buffer<def_t> _defs;
  tf::buffer<Index> _def_offsets;
  tf::offset_block_buffer<Index, Index> _edges;
  Index _n_canon = 0;
};

} // namespace tf::intersect::graph
