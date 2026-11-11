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
#include "../../../../core/buffer.hpp"
#include "../../../../exact/meta.hpp"
#include "../tet_facet_key.hpp"
#include "../tet_site_position.hpp"
#include "../tet_with_sites.hpp"
#include "./read_tet_face_cover.hpp"
#include "./tet_face_cover_scratch.hpp"
#include "./tet_face_holds_facet.hpp"
#include "./tet_projected_area2.hpp"
#include "./tet_rim_piece.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Whether `frontier` is the frontier of the cover of face `facet` read from
/// its `rim` (@ref read_tet_face_cover), sorted and each piece once;
/// whether that cover tiles the face by area exactly when it leaves no
/// frontier, each facet counted once per layer of the face lying over it
/// (@ref state_tet_cover_layers), which proves the layers; and, for a `face`
/// other than `none`, whether the face holds every facet of it. `tickets`,
/// `read` and `hashed` are the caller's scratch for the cover it reads. Every
/// site is read at the owner's deepest depth.
template <typename Owner, typename Facet>
auto check_tet_face_cover(
    const Owner &owner, const Facet &facet, const std::pair<int, int> &axes,
    const tf::buffer<tet_rim_piece<typename Owner::index_type>> &rim,
    const tf::buffer<tet_rim_piece<typename Owner::index_type>> &frontier,
    typename Owner::index_type face,
    tf::buffer<typename Owner::index_type> &tickets,
    tf::buffer<tet_rim_piece<typename Owner::index_type>> &read,
    tet_face_cover_scratch<typename Owner::index_type> &scratch) -> bool {
  using Index = typename Owner::index_type;
  return tet_with_sites(
      owner, tet_deepest_depth(owner), [&](const auto &sites) {
        using lattice = typename std::decay_t<decltype(sites)>::int_type;
        using T2 = typename tf::exact::meta<lattice>::T2;
        auto &layers = scratch.check_layers;
        read_tet_face_cover(owner, sites, facet[0], axes, rim, tickets, read,
                            layers, scratch);
        if (read.size() != frontier.size())
          return false;
        for (std::size_t i = 0; i < read.size(); ++i)
          if (read[i].lo != frontier[i].lo || read[i].hi != frontier[i].hi ||
              read[i].side != frontier[i].side ||
              (i && !tet_rim_piece_precedes(read[i - 1], read[i])))
            return false;
        T2 expected(0), covered(0);
        for (const auto &t : facet)
          expected = expected + tet_projected_area2(sites, axes, Index(t[0]),
                                                    Index(t[1]), Index(t[2]));
        for (std::size_t i = 0; i < tickets.size(); ++i) {
          const Index ticket = tickets[i];
          const auto key = tet_facet_key(
              owner._corners[std::size_t(ticket) / 4], std::size_t(ticket) % 4);
          const T2 count(layers.size() ? layers[i] : Index(1));
          covered = covered + count * tet_projected_area2(sites, axes, key[0],
                                                          key[1], key[2]);
          if (face != Owner::none && !tet_face_holds_facet(owner, ticket, face))
            return false;
        }
        return (covered == expected) == !read.size();
      });
}

} // namespace tf::topology::cdt::dt3
