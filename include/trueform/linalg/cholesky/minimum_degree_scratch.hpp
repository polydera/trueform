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
#include <cstddef>
#include <cstdint>

namespace tf::linalg {

/// What a node of the quotient graph is at one moment of the elimination.
enum class minimum_degree_kind : std::uint8_t {
  variable, ///< An uneliminated supervariable, standing for its members.
  element,  ///< An eliminated pivot, standing for the clique it formed.
  merged,   ///< A member of another node, which it is eliminated with.
  absorbed, ///< An element another element's clique covers.
  deferred  ///< A dense row, eliminated after every other.
};

/// The workspace the minimum degree walk stands on.
///
/// `lists` holds every live node's list at `start`, `length` long: a
/// variable's `elements` leading entries name the elements it touches and the
/// rest the variables it still touches directly; an element's entries name
/// the variables of its clique. The walk appends each new clique past the
/// last list and compacts the lists when the room runs out.
template <typename Index> struct minimum_degree_scratch {
  tf::buffer<Index> cursor;
  tf::buffer<Index> lists;
  tf::buffer<Index> start;
  tf::buffer<Index> length;
  tf::buffer<Index> elements;
  tf::buffer<Index> weight;
  tf::buffer<Index> degree;
  tf::buffer<tf::linalg::minimum_degree_kind> kind;
  tf::buffer<Index> head;
  tf::buffer<Index> next;
  tf::buffer<Index> last;
  tf::buffer<Index> hash_head;
  tf::buffer<Index> hash_next;
  tf::buffer<Index> hash_of;
  tf::buffer<std::size_t> mark;
  tf::buffer<Index> representative;
  tf::buffer<Index> pivots;
  tf::buffer<Index> step_of;
  tf::buffer<Index> pivot_offsets;
  tf::buffer<Index> live;
};

} // namespace tf::linalg
