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
#include "../../core/algorithm/assign_if.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"
#include "./geodesic_front_stats.hpp"
#include "./geodesic_point_state.hpp"
#include "./unfolded_distance.hpp"
#include "./unfolding_faces.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tf::geometry {

/// The relative improvement a revision must make: a corner takes an offer
/// only where its own distance exceeds the offer by more than this share of
/// the offer.
inline constexpr double geodesic_revision_tolerance = 1e-6;

/// One point the front reached, the seed whose distance it was reached at
/// (its position in the seeds), and that distance.
template <typename Index> struct geodesic_front_point {
  Index id;
  std::uint32_t seed;
  double distance;
};

/// Run the front from the points `queue` holds until it settles, or until it
/// holds more than `front_cap` points; answers whether it settled, and leaves
/// the unsettled front in `queue` otherwise.
///
/// The front is a binary heap on distance and label-correcting: the nearest
/// point is taken, and each of its faces offers each other corner the
/// distance the face states from the taken point and the face's remaining
/// corner. A corner takes an offer only where it improves its own by more
/// than @ref tf::geometry::geodesic_revision_tolerance of the new one, and
/// joins the front only below `reach`. A seed stands at zero and is never
/// revised.
///
/// An unfolding reads one source; two supports stated by different seeds
/// unfold nothing and the edge answers. Where the taken point and the
/// remaining corner share a seed the offer is
/// @ref tf::geometry::unfolded_distance from that seed; otherwise it is the
/// nearer of the two edges into the corner, from the seed of the one it
/// crosses. A point's seed is read with the stamp of the distance it
/// labelled, in the record `states` holds beside the distance, so a label never
/// outlives its distance: a label is taken only by a lower stamp, and a support
/// whose stamp is not its distance's reads as mixed and offers the edge from
/// the taken point alone. A revision moves a distance by more than a float's
/// spacing, so two distances never share a stamp. A front of one seed has
/// one source everywhere, so with `Several` false no label is read or
/// written and every support unfolds.
///
/// Distances only fall, so fronts that share `distances` lower them
/// concurrently and no improvement is lost. A point taken at a distance it
/// no longer holds was lowered since, and its lower entry does the work.
/// `polygons` carries the face membership and the vertex link. A triangle
/// mesh reads each face's image from `unfolding`, forming it the first time a
/// pop reaches the face; fronts that run at once find every image formed
/// (@ref tf::geometry::complete_unfolding_faces), so they only read it. Other
/// polygons are read from their corners. The points the front took at the
/// distance they held are counted into `stats` as one front.
template <bool Several, typename Policy, typename Index, typename Stats>
auto propagate_geodesic_distances(
    const tf::polygons<Policy> &polygons,
    tf::geometry::unfolding_faces &unfolding,
    tf::buffer<tf::geometry::geodesic_front_point<Index>> &queue,
    tf::range<tf::geometry::geodesic_point_state<Several> *, tf::dynamic_size>
        states,
    double reach, std::size_t front_cap, Stats &stats) -> bool {
  constexpr double unreached = std::numeric_limits<double>::infinity();
  const auto faces = polygons.faces();
  const auto points = polygons.points();
  const auto &membership = polygons.face_membership();
  const auto &link = polygons.vertex_link();
  const auto later = [](const tf::geometry::geodesic_front_point<Index> &x,
                        const tf::geometry::geodesic_front_point<Index> &y) {
    return x.distance > y.distance;
  };
  const auto improves = [](double reached, double held) {
    return held / reached > 1.0 + tf::geometry::geodesic_revision_tolerance;
  };

  std::size_t pops = 0;
  std::make_heap(queue.begin(), queue.end(), later);
  while (queue.size()) {
    if (queue.size() > front_cap) {
      tf::geometry::record_geodesic_front(stats, pops);
      return false;
    }
    std::pop_heap(queue.begin(), queue.end(), later);
    const Index j = queue.back().id;
    // One seed is ordinal zero everywhere.
    const std::uint32_t s_j = Several ? queue.back().seed : 0u;
    const double taken = queue.back().distance;
    queue.erase_till_end(queue.end() - 1);
    const double d_j =
        states[std::size_t(j)].distance.load(std::memory_order_relaxed);
    if (d_j < taken)
      continue;
    if constexpr (!std::is_same_v<std::decay_t<Stats>, tf::none_t>)
      ++pops;
    // An offer lands the moment a face makes it, so the faces after it read
    // the lowered point.
    const auto offer = [&](Index i, double distance, std::uint32_t seed) {
      if (states[std::size_t(i)].distance.load(std::memory_order_relaxed) ==
          0.0)
        return;
      if (tf::assign_if(states[std::size_t(i)].distance, distance, improves,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
        if constexpr (Several)
          tf::assign_if(
              states[std::size_t(i)].label,
              tf::geometry::geodesic_seed_label{seed, float(distance)},
              [](const tf::geometry::geodesic_seed_label &stated,
                 const tf::geometry::geodesic_seed_label &held) {
                return stated.stamp < held.stamp;
              },
              std::memory_order_relaxed, std::memory_order_relaxed);
        if (distance < reach) {
          queue.push_back({i, seed, distance});
          std::push_heap(queue.begin(), queue.end(), later);
        }
      }
    };
    const auto support = [&](Index v, std::uint32_t &seed, bool &labelled) {
      if constexpr (!Several) {
        seed = s_j;
        labelled = true;
        return states[std::size_t(v)].distance.load(std::memory_order_relaxed);
      } else {
        const auto &state = states[std::size_t(v)];
        const tf::geometry::geodesic_seed_label label =
            state.label.load(std::memory_order_relaxed);
        const double held = state.distance.load(std::memory_order_relaxed);
        seed = label.seed;
        labelled = float(held) == label.stamp;
        return held;
      }
    };
    // The offer at `i` from `j` and `k`, `l_ij` and `l_ik` the edges into
    // `i`.
    const auto across = [&](double unfolded, double l_ij, double l_ik,
                            double d_k, std::uint32_t s_k, bool labelled,
                            Index i) {
      if (!labelled)
        offer(i, d_j + l_ij, s_j);
      else if (s_k == s_j)
        offer(i, unfolded, s_j);
      else if (d_j + l_ij <= d_k + l_ik)
        offer(i, d_j + l_ij, s_j);
      else
        offer(i, d_k + l_ik, s_k);
    };
    if constexpr (tf::static_size_v<decltype(faces[0])> == 3) {
      for (const auto face_id : membership[j]) {
        const auto face = faces[face_id];
        auto &image = unfolding.images[std::size_t(face_id)];
        if (!unfolding.formed[std::size_t(face_id)]) {
          image = tf::geometry::make_unfolding_face(polygons[face_id]);
          unfolding.formed[std::size_t(face_id)] = char(1);
        }
        const std::size_t c_j = face[0] == j ? 0 : (face[1] == j ? 1 : 2);
        const std::size_t c_a = (c_j + 1) % 3, c_b = (c_j + 2) % 3;
        const Index a = face[c_a], b = face[c_b];
        const double l_ja = image.edges[c_j], l_ab = image.edges[c_a],
                     l_bj = image.edges[c_b];
        if constexpr (Several) {
          std::uint32_t s_b = s_j;
          bool labelled_b = false;
          const double d_b = support(b, s_b, labelled_b);
          if (d_b < unreached)
            across(labelled_b && s_b == s_j
                       ? tf::geometry::unfolded_distance(
                             l_ja, l_ab, l_bj, image.two_area, d_j, d_b)
                       : unreached,
                   l_ja, l_ab, d_b, s_b, labelled_b, a);
          std::uint32_t s_a = s_j;
          bool labelled_a = false;
          const double d_a = support(a, s_a, labelled_a);
          if (d_a < unreached)
            across(labelled_a && s_a == s_j
                       ? tf::geometry::unfolded_distance(
                             l_bj, l_ab, l_ja, image.two_area, d_j, d_a)
                       : unreached,
                   l_bj, l_ab, d_a, s_a, labelled_a, b);
        } else {
          const double d_b =
              states[std::size_t(b)].distance.load(std::memory_order_relaxed);
          if (d_b < unreached)
            offer(a,
                  tf::geometry::unfolded_distance(l_ja, l_ab, l_bj,
                                                  image.two_area, d_j, d_b),
                  s_j);
          const double d_a =
              states[std::size_t(a)].distance.load(std::memory_order_relaxed);
          if (d_a < unreached)
            offer(b,
                  tf::geometry::unfolded_distance(l_bj, l_ab, l_ja,
                                                  image.two_area, d_j, d_a),
                  s_j);
        }
      }
    } else {
      const auto p_j = points[j].template as<double>();
      const auto ring = link[j];
      for (const auto face_id : membership[j]) {
        const auto face = faces[face_id];
        for (const auto i : face) {
          // A corner across the polygon is no neighbour.
          std::size_t at = 0;
          while (at < ring.size() && ring[at] != i)
            ++at;
          if (i == j || at == ring.size())
            continue;
          const auto p_i = points[i].template as<double>();
          const double l_ij = (p_i - p_j).length();
          double nearest = unreached;
          std::uint32_t nearest_seed = s_j;
          for (const auto v : face) {
            if (v == i || v == j)
              continue;
            std::uint32_t s_v = s_j;
            bool labelled_v = false;
            const double d_v = support(v, s_v, labelled_v);
            if (!(d_v < unreached))
              continue;
            const auto p_v = points[v].template as<double>();
            double candidate = d_j + l_ij;
            std::uint32_t candidate_seed = s_j;
            if (labelled_v && s_v == s_j)
              candidate =
                  tf::geometry::unfolded_distance(p_i, p_j, p_v, d_j, d_v);
            else if (labelled_v && d_v + (p_i - p_v).length() < candidate) {
              candidate = d_v + (p_i - p_v).length();
              candidate_seed = s_v;
            }
            if (candidate < nearest) {
              nearest = candidate;
              nearest_seed = candidate_seed;
            }
          }
          if (nearest < unreached)
            offer(i, nearest, nearest_seed);
        }
      }
    }
  }
  tf::geometry::record_geodesic_front(stats, pops);
  return true;
}

} // namespace tf::geometry
