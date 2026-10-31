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
#include "../../core/aabb.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/inflated_aabb.hpp"
#include "../../core/intersects.hpp"
#include "../../core/point.hpp"
#include "../../core/transformed.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/vertex.hpp"
#include "../../spatial/search.hpp"
#include "./piece_corner_point.hpp"
#include "./reseat_pieces.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement::reseat {

/// The faces a piece meets where its source face did not: a piece stands
/// within the motion of its source, so every face it meets has a box meeting
/// the piece's own — the trees answer which, grown by the motion they cannot
/// see, and the corners decide exactly.
template <typename Index, typename Int, typename ApplyToForm, typename Reader,
          typename Created, typename Converter>
auto state_reseat_touched(const reseat_pieces<Index, Int> &pieces, Index n_tags,
                          const ApplyToForm &apply_to_form,
                          const Reader &reader, const Created &created,
                          const Converter &converter)
    -> tf::buffer<std::array<Index, 2>> {
  using pt = tf::point<Int, 3>;
  tf::buffer<std::array<Index, 2>> out;
  for (const auto &piece : pieces.pieces) {
    pt lo{}, hi{};
    std::array<tf::exact::pt3<Int>, 3> piece_points;
    for (std::size_t c = 0; c < 3; ++c) {
      const auto p =
          piece_corner_point(piece.corners[c], reader, pieces.seats, created);
      for (int k = 0; k < 3; ++k) {
        lo[k] = c == 0 ? p[k] : std::min(lo[k], p[k]);
        hi[k] = c == 0 ? p[k] : std::max(hi[k], p[k]);
      }
      piece_points[c] = tf::exact::pt3<Int>(p);
    }
    const auto box = tf::make_aabb(lo, hi);
    const auto grown = tf::inflated_aabb(box, reader.motion_bound);
    for (Index tag = 0; tag < n_tags; ++tag)
      apply_to_form(tag, [&](const auto &form) {
        const auto faces = form.faces();
        tf::search(
            form,
            [&](const auto &bv) {
              const auto b = tf::transformed(bv, tf::frame_of(form));
              return tf::intersects(tf::make_aabb(converter.convert(b.min),
                                                  converter.convert(b.max)),
                                    grown);
            },
            [&](const auto &primitive) {
              const auto id = Index(primitive.id());
              const auto face = faces[id];
              std::array<tf::exact::pt3<Int>, 3> q;
              bool outside = false;
              for (std::size_t c = 0; c < 3; ++c) {
                const auto p = reader(int(tag), Index(face[c]));
                q[c] = tf::exact::pt3<Int>(p);
              }
              for (int k = 0; k < 3 && !outside; ++k) {
                const auto flo = std::min(q[0][k], std::min(q[1][k], q[2][k]));
                const auto fhi = std::max(q[0][k], std::max(q[1][k], q[2][k]));
                outside = fhi < lo[k] || flo > hi[k];
              }
              // a plane of either triangle with the other strictly on one
              // side of it separates them
              const auto strictly_one_side = [](const auto &t, const auto &u) {
                int seen = 0;
                for (std::size_t c = 0; c < 3; ++c) {
                  const int sign =
                      tf::exact::orient3d_sign(t[0], t[1], t[2], u[c]);
                  if (sign == 0)
                    return false;
                  if (seen != 0 && sign != seen)
                    return false;
                  seen = sign;
                }
                return true;
              };
              if (!outside && !strictly_one_side(piece_points, q) &&
                  !strictly_one_side(q, piece_points))
                out.push_back({tag, id});
              return false;
            });
      });
  }
  std::sort(out.begin(), out.end());
  out.erase_till_end(std::unique(out.begin(), out.end()));
  out.erase_till_end(
      std::remove_if(out.begin(), out.end(), [&](const auto &key) {
        return std::binary_search(pieces.carved.begin(), pieces.carved.end(),
                                  key);
      }));
  return out;
}

} // namespace tf::arrangement::reseat
