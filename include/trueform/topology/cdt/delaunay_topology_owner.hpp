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
#include "../../core/buffer.hpp"
#include "../../core/index_map.hpp"
#include "./delaunay_execution_tuning.hpp"
#include "./delaunay_partition_node.hpp"
#include "./delaunay_site.hpp"
#include "./delaunay_site_partition_summary.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::topology::cdt {

/// Thin shared owner for the authoritative Delaunay carriers. RetentionPolicy
/// selects storage and post-build finalization; preparation and topology
/// construction remain shared free operations in their own headers.
template <typename Index, typename Int, typename RetentionPolicy>
struct delaunay_topology_owner {
  using index_type = Index;
  using int_type = Int;
  using site_type = delaunay_site<Int, Index>;
  using retention_policy = RetentionPolicy;
  using edge_buffer = typename RetentionPolicy::edge_buffer;
  static constexpr std::size_t k_darts_per_site_reserve =
      delaunay_execution_tuning::reserved_darts_per_site;
  static constexpr std::size_t k_max_topology_sites =
      std::size_t(std::numeric_limits<Index>::max()) / k_darts_per_site_reserve;

  auto clear_delaunay_topology() -> void {
    _sites.clear();
    _site_scratch.clear();
    _keys.clear();
    _key_scratch.clear();
    _counts.clear();
    _partition_summaries.clear();
    _partition_nodes.clear();
    _partition_leaves.clear();
    _index_map.f().clear();
    _index_map.kept_ids().clear();
    _delaunay_representative_to_unique.clear();
    _edges.clear();
    _orientation_fits_t1 = false;
    _incircle_inner_fits_t1 = false;
  }

  tf::buffer<site_type> _sites;
  tf::buffer<site_type> _site_scratch;
  tf::buffer<std::uint32_t> _keys;
  tf::buffer<std::uint32_t> _key_scratch;
  tf::buffer<std::size_t> _counts;
  tf::buffer<delaunay_site_partition_summary<Int>> _partition_summaries;
  tf::buffer<delaunay_partition_node<Index>> _partition_nodes;
  tf::buffer<std::size_t> _partition_leaves;
  /// Input-to-output identity, written in place by the preparation phase and
  /// read as the public index map. It is never filled elsewhere and moved in.
  tf::index_map_buffer<Index> _index_map;
  tf::buffer<Index> _delaunay_representative_to_unique;
  edge_buffer _edges;
  bool _orientation_fits_t1 = false;
  bool _incircle_inner_fits_t1 = false;
};

} // namespace tf::topology::cdt
