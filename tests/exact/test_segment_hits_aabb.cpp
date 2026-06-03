/**
 * @file test_segment_hits_aabb.cpp
 * @brief Tests for tf::exact::segment_hits_aabb
 *
 * Verifies the exact integer slab test for segment-vs-AABB
 * intersection across the typical configurations: contained,
 * piercing, parallel-and-missing, parallel-and-inside, endpoint
 * on a face, and fully outside.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/segment_hits_aabb.hpp>
#include <trueform/exact/vertex.hpp>

using pt3 = tf::exact::pt3<tf::exact::int32>;

TEST_CASE("Segment inside box hits", "[segment_hits_aabb]") {
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  CHECK(tf::exact::segment_hits_aabb(pt3{10, 10, 10}, pt3{90, 90, 90}, lo, hi));
  CHECK(tf::exact::segment_hits_aabb(pt3{50, 50, 50}, pt3{50, 50, 51}, lo, hi));
}

TEST_CASE("Segment piercing one face hits", "[segment_hits_aabb]") {
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // Enters through x=0 face, exits through x=100 face.
  CHECK(tf::exact::segment_hits_aabb(pt3{-50, 50, 50}, pt3{150, 50, 50}, lo, hi));
  // Diagonal through opposite corners.
  CHECK(tf::exact::segment_hits_aabb(pt3{-10, -10, -10}, pt3{110, 110, 110}, lo,
                                      hi));
}

TEST_CASE("Segment parallel and missing misses", "[segment_hits_aabb]") {
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // Parallel to x-axis, above the box in y.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{-50, 200, 50},
                                            pt3{200, 200, 50}, lo, hi));
  // Parallel to z-axis, outside in both x and y.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{-50, -50, -50},
                                            pt3{-50, -50, 200}, lo, hi));
}

TEST_CASE("Segment parallel and inside slab hits", "[segment_hits_aabb]") {
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // Parallel to x-axis, fully crossing inside the y- and z-slabs.
  CHECK(tf::exact::segment_hits_aabb(pt3{-50, 50, 50}, pt3{200, 50, 50}, lo,
                                      hi));
}

TEST_CASE("Endpoint on face hits", "[segment_hits_aabb]") {
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // Endpoint exactly on the x=0 face.
  CHECK(tf::exact::segment_hits_aabb(pt3{0, 50, 50}, pt3{-100, 50, 50}, lo,
                                      hi));
  // Endpoint exactly at a corner.
  CHECK(tf::exact::segment_hits_aabb(pt3{100, 100, 100}, pt3{200, 200, 200},
                                      lo, hi));
}

TEST_CASE("Segment fully outside misses", "[segment_hits_aabb]") {
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // Above the box, sweeping across.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{-50, 200, 50},
                                            pt3{150, 200, 50}, lo, hi));
  // Slanted line that doesn't pass through the box.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{200, 200, 200},
                                            pt3{300, 300, 300}, lo, hi));
  // Both endpoints below all slabs.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{-100, -100, -100},
                                            pt3{-50, -50, -50}, lo, hi));
}

TEST_CASE("Reversed direction on an axis", "[segment_hits_aabb]") {
  // den[i] < 0 path: b[i] < a[i] on one axis. Tests the sign-flip
  // logic in `le_ratio` when comparing across different-sign dens.
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // Hits going right-to-left along x.
  CHECK(tf::exact::segment_hits_aabb(pt3{150, 50, 50}, pt3{-50, 50, 50}, lo,
                                      hi));
  // Misses going right-to-left along x, outside in y.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{150, 200, 50},
                                            pt3{-50, 200, 50}, lo, hi));
}

TEST_CASE("Parallel on one axis + reversed on another", "[segment_hits_aabb]") {
  // Mixes the `den[i] == 0` synthetic interval (axis 2) with a real
  // negative-den axis (axis 0) and a positive-den axis (axis 1).
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  // x reversed, z constant inside slab, y crossing — hits.
  CHECK(tf::exact::segment_hits_aabb(pt3{150, -50, 50}, pt3{-50, 150, 50}, lo,
                                      hi));
  // x reversed, z constant OUTSIDE slab — misses.
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{150, -50, 200},
                                            pt3{-50, 150, 200}, lo, hi));
}

TEST_CASE("Edge graze along a box edge", "[segment_hits_aabb]") {
  // Segment runs exactly along the box's bottom edge — closed-set
  // semantics: the segment shares an entire edge with the box, so
  // the intersection is non-empty.
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  CHECK(tf::exact::segment_hits_aabb(pt3{0, 0, -50}, pt3{0, 0, 150}, lo, hi));
  CHECK(tf::exact::segment_hits_aabb(pt3{0, 50, 0}, pt3{100, 50, 0}, lo, hi));
}

TEST_CASE("Zero-length segment", "[segment_hits_aabb]") {
  // Degenerate: a == b. All three axes have den == 0. Hit iff the
  // point is inside the closed box.
  pt3 lo = {0, 0, 0}, hi = {100, 100, 100};
  CHECK(tf::exact::segment_hits_aabb(pt3{50, 50, 50}, pt3{50, 50, 50}, lo, hi));
  CHECK(tf::exact::segment_hits_aabb(pt3{0, 0, 0}, pt3{0, 0, 0}, lo, hi));
  CHECK_FALSE(tf::exact::segment_hits_aabb(pt3{-1, 50, 50}, pt3{-1, 50, 50},
                                            lo, hi));
}
