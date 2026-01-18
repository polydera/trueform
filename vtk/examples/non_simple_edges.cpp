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
#include <trueform/core.hpp>
#include <trueform/core/make_local_buffer_for_transformed.hpp>
#include <trueform/core/policy/buffer.hpp>
#include <trueform/core/policy/none.hpp>
#include <trueform/spatial.hpp>
#include <cmath>
#include <iostream>

auto test_static_polygon_transform() -> bool {
  // Test with static size polygon (triangle) - no buffer needed
  tf::points_buffer<float, 3> pts;
  pts.emplace_back(0.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 0.0f, 0.0f);
  pts.emplace_back(0.0f, 1.0f, 0.0f);

  tf::buffer<int> faces_buf;
  faces_buf.push_back(0);
  faces_buf.push_back(1);
  faces_buf.push_back(2);

  auto faces = tf::make_faces(tf::make_blocked_range<3>(tf::make_range(faces_buf)));
  auto polygons = tf::make_polygons(faces, pts.points());

  // 90 degree rotation around Z axis
  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  auto poly = polygons[0];

  // For static polygons, make_local_buffer_for_transformed returns none
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);
  static_assert(std::is_same_v<decltype(scratch), tf::none_t>,
                "Static polygons should return none_t");

  // Tag with buffer (no-op for static)
  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  // After 90 degree rotation around Z:
  // (1,0,0) -> (0,1,0)
  // (0,1,0) -> (-1,0,0)
  float eps = 1e-5f;
  bool ok = true;

  // Point 0: (0,0,0) -> (0,0,0)
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][2] - 0.0f) < eps;

  // Point 1: (1,0,0) -> (0,1,0)
  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][2] - 0.0f) < eps;

  // Point 2: (0,1,0) -> (-1,0,0)
  ok = ok && std::abs(transformed_poly[2][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[2][1] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[2][2] - 0.0f) < eps;

  return ok;
}

auto test_dynamic_polygon_transform() -> bool {
  // Test with dynamic size polygon (quad)
  tf::points_buffer<float, 3> pts;
  pts.emplace_back(0.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 1.0f, 0.0f);
  pts.emplace_back(0.0f, 1.0f, 0.0f);

  tf::buffer<int> offsets_buf;
  offsets_buf.push_back(0);
  offsets_buf.push_back(4);

  tf::buffer<int> data_buf;
  data_buf.push_back(0);
  data_buf.push_back(1);
  data_buf.push_back(2);
  data_buf.push_back(3);

  auto faces = tf::make_faces(tf::make_offset_block_range(
      tf::make_range(offsets_buf), tf::make_range(data_buf)));
  auto polygons = tf::make_polygons(faces, pts.points());

  // 90 degree rotation around Z axis
  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  auto poly = polygons[0];

  // For dynamic polygon, make_local_buffer_for_transformed returns local_buffer
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);
  static_assert(!std::is_same_v<decltype(scratch), tf::none_t>,
                "Dynamic polygon should return local_buffer");

  // Tag with buffer and transform
  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  float eps = 1e-5f;
  bool ok = true;

  // Point 0: (0,0,0) -> (0,0,0)
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][2] - 0.0f) < eps;

  // Point 1: (1,0,0) -> (0,1,0)
  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][2] - 0.0f) < eps;

  // Point 2: (1,1,0) -> (-1,1,0)
  ok = ok && std::abs(transformed_poly[2][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[2][1] - 1.0f) < eps;
  ok = ok && std::abs(transformed_poly[2][2] - 0.0f) < eps;

  // Point 3: (0,1,0) -> (-1,0,0)
  ok = ok && std::abs(transformed_poly[3][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[3][1] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[3][2] - 0.0f) < eps;

  return ok;
}

auto test_dynamic_polygon_with_frame() -> bool {
  // Test with frame (which has inverse transformation)
  tf::points_buffer<float, 3> pts;
  pts.emplace_back(0.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 1.0f, 0.0f);
  pts.emplace_back(0.0f, 1.0f, 0.0f);

  tf::buffer<int> offsets_buf;
  offsets_buf.push_back(0);
  offsets_buf.push_back(4);

  tf::buffer<int> data_buf;
  data_buf.push_back(0);
  data_buf.push_back(1);
  data_buf.push_back(2);
  data_buf.push_back(3);

  auto faces = tf::make_faces(tf::make_offset_block_range(
      tf::make_range(offsets_buf), tf::make_range(data_buf)));
  auto polygons = tf::make_polygons(faces, pts.points());

  // Create frame with 90 degree rotation around Z
  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);
  auto inv_rotation = tf::make_rotation(tf::deg<float>{-90.0f}, tf::axis<2>);
  auto frame = tf::make_frame(rotation, inv_rotation);

  // Tag with buffer for dynamic transformation
  auto poly = polygons[0];
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, frame);
  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, frame);

  float eps = 1e-5f;
  bool ok = true;

  // Same expected results as transformation test
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;

  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;

  ok = ok && std::abs(transformed_poly[2][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[2][1] - 1.0f) < eps;

  ok = ok && std::abs(transformed_poly[3][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[3][1] - 0.0f) < eps;

  return ok;
}

auto test_dynamic_polygon_no_ids() -> bool {
  // Test with dynamic size polygon without ids (using polygons_buffer)
  tf::polygons_buffer<int, float, 3, tf::dynamic_size> polys_buf;
  polys_buf.points_buffer().emplace_back(0.0f, 0.0f, 0.0f);
  polys_buf.points_buffer().emplace_back(1.0f, 0.0f, 0.0f);
  polys_buf.points_buffer().emplace_back(1.0f, 1.0f, 0.0f);
  polys_buf.points_buffer().emplace_back(0.0f, 1.0f, 0.0f);

  polys_buf.faces_buffer().offsets_buffer().push_back(0);
  polys_buf.faces_buffer().offsets_buffer().push_back(4);
  polys_buf.faces_buffer().data_buffer().push_back(0);
  polys_buf.faces_buffer().data_buffer().push_back(1);
  polys_buf.faces_buffer().data_buffer().push_back(2);
  polys_buf.faces_buffer().data_buffer().push_back(3);

  auto polygons = polys_buf.polygons();

  // 90 degree rotation around Z axis
  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  auto poly = polygons[0];

  // For dynamic polygon, make_local_buffer_for_transformed returns local_buffer
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);
  static_assert(!std::is_same_v<decltype(scratch), tf::none_t>,
                "Dynamic polygon should return local_buffer");

  // Tag with buffer and transform
  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  float eps = 1e-5f;
  bool ok = true;

  // Point 0: (0,0,0) -> (0,0,0)
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;

  // Point 1: (1,0,0) -> (0,1,0)
  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;

  // Point 2: (1,1,0) -> (-1,1,0)
  ok = ok && std::abs(transformed_poly[2][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[2][1] - 1.0f) < eps;

  // Point 3: (0,1,0) -> (-1,0,0)
  ok = ok && std::abs(transformed_poly[3][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[3][1] - 0.0f) < eps;

  return ok;
}

auto test_static_polygon_no_ids() -> bool {
  // Test with static size polygon without ids (using polygons_buffer)
  tf::polygons_buffer<int, float, 3, 3> polys_buf;
  polys_buf.points_buffer().emplace_back(0.0f, 0.0f, 0.0f);
  polys_buf.points_buffer().emplace_back(1.0f, 0.0f, 0.0f);
  polys_buf.points_buffer().emplace_back(0.0f, 1.0f, 0.0f);

  polys_buf.faces_buffer().emplace_back(0, 1, 2);

  auto polygons = polys_buf.polygons();

  // 90 degree rotation around Z axis
  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  auto poly = polygons[0];

  // For static polygon, make_local_buffer_for_transformed returns none
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);
  static_assert(std::is_same_v<decltype(scratch), tf::none_t>,
                "Static polygon should return none_t");

  // Tag with buffer (no-op for static)
  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  float eps = 1e-5f;
  bool ok = true;

  // Point 0: (0,0,0) -> (0,0,0)
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;

  // Point 1: (1,0,0) -> (0,1,0)
  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;

  // Point 2: (0,1,0) -> (-1,0,0)
  ok = ok && std::abs(transformed_poly[2][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[2][1] - 0.0f) < eps;

  return ok;
}

auto test_two_arg_tag_buffer() -> bool {
  // Test two-argument tag_buffer syntax
  tf::points_buffer<float, 3> pts;
  pts.emplace_back(0.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 1.0f, 0.0f);
  pts.emplace_back(0.0f, 1.0f, 0.0f);

  tf::buffer<int> offsets_buf;
  offsets_buf.push_back(0);
  offsets_buf.push_back(4);

  tf::buffer<int> data_buf;
  data_buf.push_back(0);
  data_buf.push_back(1);
  data_buf.push_back(2);
  data_buf.push_back(3);

  auto faces = tf::make_faces(tf::make_offset_block_range(
      tf::make_range(offsets_buf), tf::make_range(data_buf)));
  auto polygons = tf::make_polygons(faces, pts.points());

  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  auto poly = polygons[0];
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);

  // Two-argument syntax
  auto buffered_poly = tf::tag_buffer(scratch, poly);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  float eps = 1e-5f;
  bool ok = true;

  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;
  ok = ok && std::abs(transformed_poly[2][0] - (-1.0f)) < eps;
  ok = ok && std::abs(transformed_poly[3][0] - (-1.0f)) < eps;

  return ok;
}

auto test_form_local_buffer() -> bool {
  // Test that make_local_buffer_for works with form (inherits from polygons)
  tf::points_buffer<float, 3> pts;
  pts.emplace_back(0.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 1.0f, 0.0f);
  pts.emplace_back(0.0f, 1.0f, 0.0f);

  tf::buffer<int> offsets_buf;
  offsets_buf.push_back(0);
  offsets_buf.push_back(4);

  tf::buffer<int> data_buf;
  data_buf.push_back(0);
  data_buf.push_back(1);
  data_buf.push_back(2);
  data_buf.push_back(3);

  auto faces = tf::make_faces(tf::make_offset_block_range(
      tf::make_range(offsets_buf), tf::make_range(data_buf)));
  auto polygons = tf::make_polygons(faces, pts.points());

  // Build an empty tree (we just need it for form creation)
  tf::aabb_tree<int, float, 3> tree;
  tree.build(polygons, tf::config_tree(4, 4));

  // Create form
  auto form = polygons | tf::tag(tree);

  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  // Verify we can tag and transform
  auto poly = form[0];

  // make_local_buffer_for_transformed should work with form (dynamic polygons)
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);
  static_assert(!std::is_same_v<decltype(scratch), tf::none_t>,
                "Form with dynamic polygons should return local_buffer");

  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  float eps = 1e-5f;
  bool ok = true;

  // Point 0: (0,0,0) -> (0,0,0)
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;

  // Point 1: (1,0,0) -> (0,1,0)
  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;

  return ok;
}

auto test_static_form_local_buffer() -> bool {
  // Test that make_local_buffer_for works with form (static triangles)
  tf::points_buffer<float, 3> pts;
  pts.emplace_back(0.0f, 0.0f, 0.0f);
  pts.emplace_back(1.0f, 0.0f, 0.0f);
  pts.emplace_back(0.0f, 1.0f, 0.0f);

  tf::buffer<int> faces_buf;
  faces_buf.push_back(0);
  faces_buf.push_back(1);
  faces_buf.push_back(2);

  auto faces = tf::make_faces(tf::make_blocked_range<3>(tf::make_range(faces_buf)));
  auto polygons = tf::make_polygons(faces, pts.points());

  // Build tree
  tf::aabb_tree<int, float, 3> tree;
  tree.build(polygons, tf::config_tree(4, 4));

  // Create form
  auto form = polygons | tf::tag(tree);

  auto rotation = tf::make_rotation(tf::deg<float>{90.0f}, tf::axis<2>);

  // Verify we can still tag (no-op) and transform
  auto poly = form[0];

  // make_local_buffer_for_transformed should return none for static polygons
  auto scratch = tf::core::make_local_buffer_for_transformed(poly, rotation);
  static_assert(std::is_same_v<decltype(scratch), tf::none_t>,
                "Form with static triangles should return none_t");

  auto buffered_poly = poly | tf::tag(scratch);
  auto transformed_poly = tf::transformed(buffered_poly, rotation);

  float eps = 1e-5f;
  bool ok = true;

  // Point 0: (0,0,0) -> (0,0,0)
  ok = ok && std::abs(transformed_poly[0][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[0][1] - 0.0f) < eps;

  // Point 1: (1,0,0) -> (0,1,0)
  ok = ok && std::abs(transformed_poly[1][0] - 0.0f) < eps;
  ok = ok && std::abs(transformed_poly[1][1] - 1.0f) < eps;

  return ok;
}

int main() {
  int failed = 0;

  std::cout << "test_static_polygon_transform: ";
  if (test_static_polygon_transform()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_dynamic_polygon_transform: ";
  if (test_dynamic_polygon_transform()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_dynamic_polygon_with_frame: ";
  if (test_dynamic_polygon_with_frame()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_dynamic_polygon_no_ids: ";
  if (test_dynamic_polygon_no_ids()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_static_polygon_no_ids: ";
  if (test_static_polygon_no_ids()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_two_arg_tag_buffer: ";
  if (test_two_arg_tag_buffer()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_form_local_buffer: ";
  if (test_form_local_buffer()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  std::cout << "test_static_form_local_buffer: ";
  if (test_static_form_local_buffer()) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    ++failed;
  }

  return failed;
}
