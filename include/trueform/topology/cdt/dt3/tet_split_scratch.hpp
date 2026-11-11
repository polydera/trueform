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
#include "../../../core/buffer.hpp"
#include "./tet_facet_owner.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A facet of the cells a split replaces that the split does not divide:
/// its `key`, the cell `across` it and that cell's slot `back` onto it,
/// with the protection it holds (@ref split_tetrahedralization_site).
template <typename Index> struct tet_split_facet {
  std::array<Index, 3> key;
  Index across;
  std::size_t back;
  bool held;
  Index owner;
};

/// A facet the split divides, each half inheriting its protection.
template <typename Index> struct tet_split_half {
  std::array<Index, 3> key;
  bool held;
  Index owner;
};

/// A protection the split seats once every link of its cells stands: the
/// `slot` of `row` and the face that owns the facet there.
template <typename Index> struct tet_split_seat {
  Index row;
  std::size_t slot;
  Index owner;
};

/// The lists one split of the simplex a site lies in walks on: `old` the
/// cells it replaces and `made` the cells it writes, `halves` the facets it
/// divides, `outer` the facets it carries across, `rows` the rows it writes
/// them in, `protect` the protections it seats and `owners` the stack rows a
/// shared half inherits, each face with its layers (@ref
/// split_tetrahedralization_site).
template <typename Index> struct tet_split_scratch {
  tf::buffer<Index> old;
  tf::buffer<std::array<Index, 4>> made;
  tf::buffer<tet_split_half<Index>> halves;
  tf::buffer<tet_split_facet<Index>> outer;
  tf::buffer<Index> rows;
  tf::buffer<tet_split_seat<Index>> protect;
  tf::buffer<tet_facet_owner<Index>> owners;
};

} // namespace tf::topology::cdt::dt3
