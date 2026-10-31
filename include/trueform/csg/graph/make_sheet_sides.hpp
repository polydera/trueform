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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./domain_inclusions.hpp"
#include "./find_sheet_rim_volumes.hpp"
#include "./find_sheet_rims.hpp"
#include "./sheet_sides.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The sheet sides the volumes state — each volume's inside split
///        into the components a sheet's walls bound — every other pair
///        unstated. Without sheets the table holds no state at all.
///
/// Reads the volume columns of `inc`, which are final. A nesting merge
/// joins two false halves of one region, so it joins them here too, and a
/// free rim of the sheet joins the two domains it stands between.
template <typename Index, typename OpenMask>
auto make_sheet_sides(
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::component_crossings<Index> &crossings,
    const tf::offset_block_buffer<Index, Index> &domain_components,
    const tf::buffer<std::array<Index, 2>> &nesting_merges,
    const tf::csg::graph::domain_inclusions &inc, const OpenMask &open_mask,
    Index n_tags, const tf::buffer<char> &is_sheet_tag)
    -> sheet_sides<Index> {
  using sides_t = sheet_sides<Index>;
  sides_t out;
  out.n_domains = desc.n_domains;
  out.sheet_of_tag.allocate(static_cast<std::size_t>(n_tags));
  tf::buffer<Index> volumes;
  for (Index t = Index(0); t < n_tags; ++t) {
    const bool sheet = t < Index(is_sheet_tag.size()) && is_sheet_tag[t];
    out.sheet_of_tag[t] = sheet ? Index(out.tags.size()) : Index(-1);
    if (sheet)
      out.tags.push_back(t);
    else
      volumes.push_back(t);
  }
  if (out.tags.size() == 0)
    return out;
  const Index n_domains = desc.n_domains;
  out.states.allocate(out.tags.size() * static_cast<std::size_t>(n_domains));
  tf::parallel_fill(out.states, sides_t::unstated);
  const Index n_volumes = Index(volumes.size());
  if (n_volumes == Index(0) || n_domains == Index(0))
    return out;
  const auto rims =
      tf::csg::graph::find_sheet_rims(out, desc, crossings, open_mask);
  out.rim_volumes = tf::csg::graph::find_sheet_rim_volumes(out, rims, inc);
  tf::buffer<std::array<Index, 3>> rim_peers; // (sheet, domain, domain)
  for (const auto &rim : rims) {
    rim_peers.push_back({rim[0], rim[1], rim[2]});
    rim_peers.push_back({rim[0], rim[2], rim[1]});
  }
  std::sort(rim_peers.begin(), rim_peers.end());

  tf::buffer<std::array<Index, 2>> peers; // both directions, by domain
  for (const auto &m : nesting_merges) {
    peers.push_back({m[0], m[1]});
    peers.push_back({m[1], m[0]});
  }
  std::sort(peers.begin(), peers.end());

  // a domain's stamp is the (sheet, volume) pair that last reached it, so
  // the block's scratch is never cleared
  struct local_t {
    tf::buffer<Index> stamp;
    tf::buffer<Index> members;
  };
  tf::buffer<std::array<Index, 3>> stated; // (sheet, domain, side)
  tf::generic_generate(
      tf::make_sequence_range(out.n_sheets() * n_volumes), stated,
      [&](Index pair, tf::buffer<std::array<Index, 3>> &buffer,
          local_t &local) {
        if (local.stamp.size() != std::size_t(n_domains)) {
          local.stamp.allocate(static_cast<std::size_t>(n_domains));
          for (auto &s : local.stamp)
            s = Index(-1);
        }
        const Index k = pair / n_volumes;
        const std::size_t t =
            static_cast<std::size_t>(volumes[pair % n_volumes]);
        for (Index first = Index(0); first < n_domains; ++first) {
          if (local.stamp[first] == pair || !inc.test(std::size_t(first), t))
            continue;
          local.stamp[first] = pair;
          local.members.clear();
          local.members.push_back(first);
          char side = sides_t::unstated;
          auto reach = [&](Index other) {
            if (local.stamp[other] == pair || !inc.test(std::size_t(other), t))
              return;
            local.stamp[other] = pair;
            local.members.push_back(other);
          };
          for (std::size_t head = 0; head < local.members.size(); ++head) {
            const Index d = local.members[head];
            const std::array<Index, 2> key{d, Index(0)};
            for (auto it = std::lower_bound(peers.begin(), peers.end(), key);
                 it != peers.end() && (*it)[0] == d; ++it)
              reach((*it)[1]);
            const std::array<Index, 3> rim_key{k, d, Index(0)};
            for (auto it = std::lower_bound(rim_peers.begin(), rim_peers.end(),
                                            rim_key);
                 it != rim_peers.end() && (*it)[0] == k && (*it)[1] == d;
                 ++it)
              reach((*it)[2]);
            for (const Index c : domain_components[std::size_t(d)]) {
              bool on_sheet = false;
              out.for_each_sheet_layer(
                  desc, crossings, c, [&](Index j, Index behind_side) {
                    if (j != k)
                      return;
                    on_sheet = true;
                    if (desc.domain_of_side[2 * c + behind_side] == d)
                      side = char(side | sides_t::behind);
                    if (desc.domain_of_side[2 * c + 1 - behind_side] == d)
                      side = char(side | sides_t::front);
                  });
              if (on_sheet)
                continue;
              const Index d0 = desc.domain_of_side[2 * c + 0];
              reach(d0 == d ? desc.domain_of_side[2 * c + 1] : d0);
            }
          }
          if (side != sides_t::unstated)
            for (const Index d : local.members)
              buffer.push_back({k, d, Index(side)});
        }
      },
      local_t{});

  tbb::parallel_sort(stated.begin(), stated.end());
  for (std::size_t lo = 0; lo < stated.size();) {
    std::size_t hi = lo;
    char one_side = sides_t::unstated;
    bool wraps = false;
    while (hi < stated.size() && stated[hi][0] == stated[lo][0] &&
           stated[hi][1] == stated[lo][1]) {
      const char s = char(stated[hi][2]);
      if (s == sides_t::wraps)
        wraps = true;
      else
        one_side = char(one_side | s);
      ++hi;
    }
    if (one_side == sides_t::wraps)
      ++out.n_conflicts;
    out.state(stated[lo][0], stated[lo][1]) =
        one_side != sides_t::unstated ? one_side
        : wraps                       ? sides_t::wraps
                                      : sides_t::unstated;
    lo = hi;
  }
  return out;
}

} // namespace tf::csg::graph
