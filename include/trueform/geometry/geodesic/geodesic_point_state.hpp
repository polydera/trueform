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
#include <atomic>
#include <cstdint>

namespace tf::geometry {

/// The seed that stated a point's distance, and the stamp of that distance:
/// one record, so a label never outlives its distance.
struct geodesic_seed_label {
  std::uint32_t seed;
  float stamp;
};

static_assert(
    std::atomic<tf::geometry::geodesic_seed_label>::is_always_lock_free,
    "a seed label is one lock-free record");

/// What the front holds of one point, chosen by whether the seeds are
/// several: the distance alone, or the distance with its seed label beside
/// it, so a support is read from one record.
template <bool Several> struct geodesic_point_state;

template <> struct geodesic_point_state<false> {
  std::atomic<double> distance;

  explicit geodesic_point_state(double held) : distance(held) {}
};

template <> struct alignas(16) geodesic_point_state<true> {
  std::atomic<double> distance;
  std::atomic<tf::geometry::geodesic_seed_label> label;

  geodesic_point_state(double held, tf::geometry::geodesic_seed_label stated)
      : distance(held), label(stated) {}
};

static_assert(sizeof(tf::geometry::geodesic_point_state<false>) == 8,
              "one seed holds the distance alone");
static_assert(sizeof(tf::geometry::geodesic_point_state<true>) == 16,
              "several seeds hold the distance and its label in one record");

} // namespace tf::geometry
