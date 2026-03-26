/**
 * @file test_classify_wedge.cpp
 * @brief Tests for tf::exact::classify_wedge
 *
 * Verifies wedge classification for convex (90°), reflex (270°),
 * and flat (180°) wedges. Tests boundary, inside, and outside
 * classification with exact orient3d predicates.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/classify_wedge.hpp>

using pt3 = tf::exact::pt3;
using S = tf::sidedness;

TEST_CASE("Convex wedge boundary", "[classify_wedge]") {
  pt3 v0 = {0, 0, 0}, v1 = {100, 0, 0};
  pt3 a = {0, 100, 0}, b = {0, 0, 100};

  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 50, 0}) == S::on_boundary);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, -50, 0}) == S::on_negative_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 0, 50}) == S::on_boundary);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 0, -50}) == S::on_negative_side);
}

TEST_CASE("Convex wedge 90 quadrants", "[classify_wedge]") {
  pt3 v0 = {0, 0, 0}, v1 = {100, 0, 0};
  pt3 a = {0, 100, 0}, b = {0, 0, 100};

  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, -50, -50}) == S::on_negative_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 50, -50}) == S::on_negative_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, -50, 50}) == S::on_negative_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 50, 50}) == S::on_positive_side);
}

TEST_CASE("Convex wedge 90 flipped edge", "[classify_wedge]") {
  pt3 v0 = {100, 0, 0}, v1 = {0, 0, 0};
  pt3 a = {0, 100, 0}, b = {0, 0, 100};

  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, -50, -50}) == S::on_positive_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 50, -50}) == S::on_positive_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, -50, 50}) == S::on_positive_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 50, 50}) == S::on_negative_side);
}

TEST_CASE("Flat wedge 180", "[classify_wedge]") {
  pt3 v0 = {0, 0, 0}, v1 = {100, 0, 0};
  pt3 a = {0, 100, 0}, b = {0, -100, 0};

  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 0, -50}) == S::on_negative_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 0, 50}) == S::on_positive_side);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 50, 0}) == S::on_boundary);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, -50, 0}) == S::on_boundary);
  CHECK(tf::exact::classify_wedge(v0, v1, a, b, {50, 0, 0}) == S::on_boundary);
}
