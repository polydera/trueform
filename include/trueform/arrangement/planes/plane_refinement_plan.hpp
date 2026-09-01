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
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_refinement_physical_split.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

inline constexpr std::uint8_t plane_refinement_immutable_source = 0;
inline constexpr std::uint8_t plane_refinement_current_source = 1;

/// One accepted refiner choice on the exact generation-input piece it saw.
/// `feature` is the ordered flat endpoint pair and `parameter` is in that
/// piece's own key-order frame. `inverted` says the two orders disagree, so a
/// consumer that binds the parameter to the NAME states `whole - parameter`.
/// `source` says which of the two prepared tables owns `group`; the wave
/// consumes that `(source, group)` currency verbatim.
template <typename Index, typename Int> struct plane_refinement_split {
  using param_t = typename tf::exact::meta<Int>::param_type;

  std::array<Index, 2> feature;
  Index plane;
  Index group;
  param_t parameter;
  std::uint8_t source;
  std::uint8_t inverted;
};

struct plane_refinement_census {
  std::size_t planes = 0;
  std::size_t raw_boundary_proposals = 0;
  std::size_t unique_boundary_splits = 0;
  std::size_t ring_boundary_splits = 0;
  std::size_t physical_delivery_rows = 0;
  std::size_t rings_run = 0;
  std::size_t affected_planes = 0;
  std::size_t promoted_planes = 0;
  std::size_t steiners = 0;
  std::size_t projection_weld_relations = 0;
  std::size_t refused_planes = 0;
};

/// The owning, one-use result of refinement discovery. It is proposal and
/// scheduling data, not a second identity or ancestry arena. Boundary rows are
/// consumed into PA's ordinary wave; exact Steiner points are direct forms in
/// owner-aligned blocks and never enter topology propagation. A plan entering
/// the explicit rvalue consumer names the pristine generation: `splits` and
/// `steiners` are producer facts, while `physical_splits` and `promoted_faces`
/// must still be empty consumer scratch. Every split belongs to its stated
/// plane and is already snapped, sorted and separated exactly as discovery
/// emits it. `welds` are the exact input-index-map product consumed by PA's
/// ordinary weld barrier. A refused plane owns no split or Steiner product,
/// but may own weld evidence prepared before constraint recovery refused.
template <typename Index, typename Int> struct plane_refinement_plan {
  tf::buffer<plane_refinement_split<Index, Int>> splits;
  tf::buffer<plane_refinement_physical_split<Index, Int>> physical_splits;
  tf::offset_block_buffer<Index, tf::exact::pt3<Int>> steiners;
  tf::buffer<plane_flat_weld<Index>> welds;
  tf::buffer<Index> refused_planes;
  tf::buffer<Index> promoted_faces;
  plane_refinement_census census;

  auto empty() const -> bool {
    return splits.size() == 0 && physical_splits.size() == 0 &&
           steiners.data_buffer().size() == 0 && welds.size() == 0 &&
           refused_planes.size() == 0 && promoted_faces.size() == 0;
  }
};

} // namespace tf::arrangement
