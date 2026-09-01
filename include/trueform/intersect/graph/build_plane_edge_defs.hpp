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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../exact/meta.hpp"
#include "../../topology/topo_type.hpp"
#include "../records/tagged_intersection.hpp"
#include "./edge.hpp"
#include "./extract_plane_edges.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_pair_carrier.hpp"
#include "./plane_point_generator.hpp"
#include "./vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::intersect::graph {

/// Every face's edge definitions, its contour tag, and its claims on
/// the coplanar contacts it takes part in.
///
/// A definition names its own face, base-loop position and original
/// edge, so a face's span of the emission table is its whole edge set
/// and `face_def_offsets` is the reverse lookup — the aggregation runs
/// in input-block order, which is what lets the emission ids rebase
/// against one running offset.
///
/// The carrier is the descriptor table — every face a point was
/// delivered to — and `subranges` are that face's pair groups at the
/// same block position. A face whose pairs all proved inert carries an
/// empty subrange and still states its whole base loop.
template <typename Index, typename Int, typename Descriptors,
          typename Subranges, typename AllLoops, typename Ibp,
          typename ApplyToFace, typename GetPoint, typename SideNonManifold>
auto build_plane_edge_defs(const Descriptors &descriptors,
                           const Subranges &subranges,
                           const AllLoops &all_loops, const Ibp &ibp,
                           const ApplyToFace &apply_to_face,
                           const GetPoint &get_point,
                           const SideNonManifold &side_non_manifold,
                           tf::buffer<plane_edge_def<Index>> &edge_defs,
                           tf::buffer<std::int16_t> &face_contour,
                           tf::buffer<Index> &face_def_offsets)
    -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using def_t = plane_edge_def<Index>;
  using vertex_t = vertex<Index>;
  using vsource = vertex_source;
  const auto vertex_offsets = ibp.vertex_offsets();
  const auto n = descriptors.size();
  face_def_offsets.allocate(n + 1);
  face_def_offsets[0] = 0;
  std::size_t face_i = 1;

  // {LOOP-ORDER endpoint pair, loop position}: sorted by value, so the
  // duplicates of one DIRECTED wall are adjacent and the earliest
  // position states it. AB unions with AB and BA with BA, never across —
  // the two directions are oppositely wound and are two facts.
  struct loop_key_t {
    std::array<Index, 4> key;
    Index position;
  };

  struct local_t {
    tf::buffer<Index> counts;
    tf::buffer<def_t> data;
    tf::buffer<std::int16_t> covered;
    tf::buffer<std::int16_t> ctags;
    tf::buffer<edge<Index>> raw;
    tf::buffer<chain_node<Index, Int, tf::intersect::tagged_intersection<Index>>>
        chain;
    tf::small_vector<loop_key_t, 32> loop_keys;
    tf::small_vector<char, 32> loop_keep;
  };

  auto task = [&](auto &&range, local_t &local) {
    auto apply_to_face_f = apply_to_face;
    auto get_point_f = get_point;
    local.counts.allocate(range.size());
    auto cit = local.counts.begin();
    for (auto &&[this_loop_idx, descriptor] : range) {
      const auto old_size = local.data.size();
      {
        auto &&subrange = subranges[this_loop_idx];
        const auto face_group = Index(this_loop_idx);
        const auto tag = descriptor.tag;
        const auto object = descriptor.object;
        auto push_def = [&](std::array<Index, 2> a, std::array<Index, 2> b,
                            std::int16_t tag_other, Index object_other,
                            std::int16_t ordinal, std::int16_t side, bool whole,
                            bool fan, int radial = 0,
                            bool non_manifold = false) {
          std::uint8_t flags =
              whole ? plane_edge_whole_side_flag : std::uint8_t(0);
          if (fan)
            flags = std::uint8_t(flags | plane_edge_fan_flag);
          if (non_manifold)
            flags = std::uint8_t(flags | plane_edge_non_manifold_flag);
          if (radial != 0)
            flags = std::uint8_t(flags | plane_edge_radial_flag |
                                 (radial < 0 ? plane_edge_radial_reversed_flag
                                             : std::uint8_t(0)));
          if (b < a) {
            std::swap(a, b);
            flags = std::uint8_t(flags | plane_edge_reversed_flag);
          }
          local.data.push_back({a[1], b[1], Index(local.data.size()), face_group,
                                object_other, std::int16_t(a[0]),
                                std::int16_t(b[0]), tag_other, ordinal, side,
                                flags});
        };
        apply_to_face_f(
            tag, object,
            [&, &this_loop_idx = this_loop_idx](const auto &face) {
              const std::size_t face_size = face.size();
              local.raw.clear();
              local.covered.clear();
              extract_plane_edges<Index, Int>(
                  subrange, Index(face_size), this_loop_idx, all_loops,
                  descriptors, apply_to_face_f, get_point_f, local.chain,
                  local.raw, local.covered);
              // the radial carrier orientation of an interior cut: the
              // producing pair is in hand exactly here, and the answer
              // comes off the endpoints' generators — never off a
              // materialized position
              std::array<T1, 3> a1{}, a2{};
              const bool own_plane = plane_face_basis<Int>(
                  face,
                  [&](std::size_t k) { return get_point_f(tag, Index(face[k])); },
                  a1, a2);
              plane_pair_carrier<Int> carrier;
              std::int16_t carrier_tag = -1;
              Index carrier_object = Index(-1);
              bool carrier_named = false;
              for (const auto &r : local.raw) {
                if (!carrier_named || r.tag_other != carrier_tag ||
                    r.object_other != carrier_object) {
                  carrier_tag = r.tag_other;
                  carrier_object = r.object_other;
                  carrier_named = true;
                  carrier = plane_pair_carrier<Int>{};
                  if (own_plane)
                    apply_to_face_f(
                        carrier_tag, carrier_object, [&](const auto &partner) {
                          std::array<T1, 3> b1{}, b2{};
                          if (plane_face_basis<Int>(
                                  partner,
                                  [&](std::size_t k) {
                                    return get_point_f(carrier_tag,
                                                       Index(partner[k]));
                                  },
                                  b1, b2))
                            carrier =
                                make_plane_pair_carrier<Int>(a1, a2, b1, b2);
                        });
                }
                int radial = 0;
                if (carrier.valid) {
                  const auto low = r.point_0 < r.point_1 ? r.point_0 : r.point_1;
                  const auto high =
                      r.point_0 < r.point_1 ? r.point_1 : r.point_0;
                  radial = plane_edge_carrier_sign<Index, Int>(
                      carrier,
                      plane_point_generator_of<Index, Int>(
                          ibp, vertex_offsets, get_point_f, low),
                      plane_point_generator_of<Index, Int>(
                          ibp, vertex_offsets, get_point_f, high));
                }
                push_def({Index(-1), r.point_0}, {Index(-1), r.point_1},
                         r.tag_other, r.object_other, std::int16_t(-1),
                         std::int16_t(-1), false, true, radial);
              }
              // The entire base loop enters wholesale as ordinary boundary
              // edges (ordinal = loop position); the extractor already
              // rejects everything the loop states, so the only reduction
              // left is the loop against itself — and it is a reduction of
              // ONE DIRECTED WALL.
              //
              // A loop may walk one wall twice. It walks it the OTHER way
              // when it does: an antenna states its spike out and back, and
              // a corner whose two neighbouring identities a tolerance band
              // welded into one pinches the ring the same way. AB and BA are
              // oppositely wound and are two instances of one canonical
              // group — groups fuse, instances never do — so the key that
              // reduces here carries the winding and only AB meets AB.
              //
              // Both instances stand, and their multiplicity IS the wall's
              // parity, which @ref
              // tf::arrangement::prepare_plane_triangulation reads: a member
              // restating a row toggles it, so the ring passes straight
              // through the pinch and the spike bounds nothing. Reduce the
              // pair to one row and that wall stands alone — a boundary
              // chain with a dead end, whose region walk covers less than
              // the face, in silence.
              auto loop = all_loops[this_loop_idx];
              const auto m = loop.size();
              auto vk = [&](const vertex_t &v) -> std::array<Index, 2> {
                return v.source == vsource::created
                           ? std::array<Index, 2>{Index(-1), v.id}
                           : std::array<Index, 2>{Index(tag), v.id};
              };
              local.loop_keys.clear();
              local.loop_keep.assign(m, char(1));
              for (std::size_t j = 0; j < m; ++j) {
                const auto ka = vk(loop[j]);
                const auto kb = vk(loop[(j + 1) % m]);
                if (ka == kb) {
                  local.loop_keep[j] = 0;
                  continue;
                }
                local.loop_keys.push_back(
                    {{ka[0], ka[1], kb[0], kb[1]}, Index(j)});
              }
              std::sort(local.loop_keys.begin(), local.loop_keys.end(),
                        [](const loop_key_t &x, const loop_key_t &y) {
                          return std::tie(x.key, x.position) <
                                 std::tie(y.key, y.position);
                        });
              for (std::size_t k = 1; k < local.loop_keys.size(); ++k)
                if (local.loop_keys[k].key == local.loop_keys[k - 1].key)
                  local.loop_keep[std::size_t(local.loop_keys[k].position)] = 0;
              for (std::size_t j = 0; j < m; ++j) {
                if (!local.loop_keep[j])
                  continue;
                const auto &a = loop[j];
                const auto &b = loop[(j + 1) % m];
                // the side the loop position lies on: whichever endpoint
                // states an edge, else the vertex the side starts at
                auto side = a.sub_id.label == tf::topo_type::edge
                                ? std::int16_t(a.sub_id.id)
                                : (b.sub_id.label == tf::topo_type::edge
                                       ? std::int16_t(b.sub_id.id)
                                       : std::int16_t(a.sub_id.id));
                bool whole = false;
                if (side >= 0 && std::size_t(side) < face_size) {
                  const auto next = (std::size_t(side) + 1) % face_size;
                  whole = a.sub_id.label == tf::topo_type::vertex &&
                          b.sub_id.label == tf::topo_type::vertex &&
                          std::size_t(a.sub_id.id) == std::size_t(side) &&
                          std::size_t(b.sub_id.id) == next;
                } else {
                  side = std::int16_t(-1);
                }
                bool fan = false;
                for (const auto c : local.covered)
                  fan = fan || std::size_t(c) == j;
                // the mesh's own non-manifold edge is an input fact,
                // stated by the form's edge link, carried on the side
                const bool non_manifold =
                    side >= 0 && side_non_manifold(tag, object, side);
                push_def(vk(a), vk(b), std::int16_t(tag), object,
                         std::int16_t(j), side, whole, fan, 0, non_manifold);
              }
            });
      }
      std::int16_t ct = -1;
      for (auto e = old_size; e < local.data.size(); ++e) {
        const auto &d = local.data[e];
        if (d.ordinal != std::int16_t(-1))
          continue;
        if (ct == std::int16_t(-1))
          ct = d.tag_other;
        else if (ct != d.tag_other) {
          ct = -2;
          break;
        }
      }
      local.ctags.push_back(ct);
      *cit++ = Index(local.data.size() - old_size);
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.ctags, face_contour);
    const auto old_size = edge_defs.size();
    edge_defs.reallocate(old_size + local.data.size());
    const auto offset = Index(old_size);
    auto it = edge_defs.begin() + old_size;
    for (auto d : local.data) {
      d.id += offset;
      *it++ = d;
    }
    for (auto sz : local.counts) {
      face_def_offsets[face_i] = face_def_offsets[face_i - 1] + sz;
      ++face_i;
    }
  };

  tf::blocked_reduce_sequenced_aggregate(tf::enumerate(descriptors), tf::none,
                                         local_t{}, task, agg);
}

} // namespace tf::intersect::graph
