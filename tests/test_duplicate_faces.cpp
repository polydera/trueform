/*
 * Duplicate faces detection test - verifies detection of duplicate polygons
 */
#include <iostream>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/topology/compute_unique_faces_mask.hpp>
#include <trueform/trueform.hpp>

// ============================================================================
// Test infrastructure
// ============================================================================

struct test_result {
  bool passed = true;
  std::string failures;

  auto fail(const std::string &msg) -> void {
    passed = false;
    failures += "  " + msg + "\n";
  }
};

// ============================================================================
// Helper to build polygons_buffer
// ============================================================================

template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
auto make_test_buffer(std::initializer_list<tf::point<RealT, Dims>> pts,
                      std::initializer_list<std::array<Index, Ngon>> fcs) {
  tf::polygons_buffer<Index, RealT, Dims, Ngon> buffer;
  for (auto &p : pts)
    buffer.points_buffer().push_back(p);
  for (auto &f : fcs)
    buffer.faces_buffer().push_back(f);
  return buffer;
}

template <typename Index, typename RealT, std::size_t Dims>
auto make_test_buffer_dynamic(
    std::initializer_list<tf::point<RealT, Dims>> pts,
    std::initializer_list<std::initializer_list<Index>> fcs) {
  tf::polygons_buffer<Index, RealT, Dims, tf::dynamic_size> buffer;
  for (auto &p : pts)
    buffer.points_buffer().push_back(p);
  for (auto &f : fcs)
    buffer.faces_buffer().push_back(f);
  return buffer;
}

// ============================================================================
// Triangle tests (static size = 3)
// ============================================================================

auto test_triangles_no_duplicates() -> test_result {
  test_result result;

  auto buffer = make_test_buffer<int, float, 3, 3>(
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.5f, 1.f, 0.f}, {0.5f, 0.5f, 1.f}},
      {{{0, 1, 2}}, {{0, 1, 3}}, {{1, 2, 3}}, {{0, 2, 3}}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  for (std::size_t i = 0; i < mask.size(); ++i) {
    if (!mask[i]) {
      result.fail("Face " + std::to_string(i) +
                  " marked as duplicate but shouldn't be");
    }
  }

  return result;
}

auto test_triangles_identical_duplicate() -> test_result {
  test_result result;

  auto buffer = make_test_buffer<int, float, 3, 3>(
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.5f, 1.f, 0.f}},
      {{{0, 1, 2}}, {{0, 1, 2}}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique (has smallest ID)");
  if (mask[1])
    result.fail("Face 1 should be marked as duplicate");

  return result;
}

auto test_triangles_reversed_duplicate() -> test_result {
  test_result result;

  auto buffer = make_test_buffer<int, float, 3, 3>(
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.5f, 1.f, 0.f}},
      {{{0, 1, 2}}, {{0, 2, 1}}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique");
  if (mask[1])
    result.fail("Face 1 (reversed) should be marked as duplicate");

  return result;
}

auto test_triangles_rotated_duplicate() -> test_result {
  test_result result;

  auto buffer = make_test_buffer<int, float, 3, 3>(
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.5f, 1.f, 0.f}},
      {{{0, 1, 2}}, {{1, 2, 0}}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique");
  if (mask[1])
    result.fail("Face 1 (rotated) should be marked as duplicate");

  return result;
}

auto test_triangles_multiple_duplicates() -> test_result {
  test_result result;

  auto buffer = make_test_buffer<int, float, 3, 3>(
      {{0.f, 0.f, 0.f},
       {1.f, 0.f, 0.f},
       {0.5f, 1.f, 0.f},
       {2.f, 0.f, 0.f},
       {1.5f, 1.f, 0.f}},
      {{{0, 1, 2}}, {{1, 3, 4}}, {{0, 1, 2}}, {{3, 4, 1}}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique");
  if (!mask[1])
    result.fail("Face 1 should be unique");
  if (mask[2])
    result.fail("Face 2 should be marked as duplicate of 0");
  if (mask[3])
    result.fail("Face 3 should be marked as duplicate of 1");

  return result;
}

// ============================================================================
// Dynamic size polygon tests
// ============================================================================

auto test_dynamic_quad_rotated() -> test_result {
  test_result result;

  auto buffer = make_test_buffer_dynamic<int, float, 3>(
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
      {{0, 1, 2, 3}, {2, 3, 0, 1}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique");
  if (mask[1])
    result.fail("Face 1 (rotated) should be marked as duplicate");

  return result;
}

auto test_dynamic_quad_reversed() -> test_result {
  test_result result;

  auto buffer = make_test_buffer_dynamic<int, float, 3>(
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
      {{0, 1, 2, 3}, {0, 3, 2, 1}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique");
  if (mask[1])
    result.fail("Face 1 (reversed) should be marked as duplicate");

  return result;
}

auto test_dynamic_pentagon_rotated() -> test_result {
  test_result result;

  auto buffer = make_test_buffer_dynamic<int, float, 3>(
      {{0.f, 0.f, 0.f},
       {1.f, 0.f, 0.f},
       {1.3f, 0.8f, 0.f},
       {0.5f, 1.2f, 0.f},
       {-0.3f, 0.8f, 0.f}},
      {{0, 1, 2, 3, 4}, {3, 4, 0, 1, 2}});

  auto polygons = buffer.polygons();
  tf::face_membership<int> fmem(polygons);

  tf::buffer<bool> mask;
  mask.allocate(buffer.size());
  tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

  if (!mask[0])
    result.fail("Face 0 should be unique");
  if (mask[1])
    result.fail("Face 1 (rotated pentagon) should be marked as duplicate");

  return result;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "Running duplicate faces detection tests...\n\n";

  auto run_test = [](const char *name, auto test_fn) {
    auto result = test_fn();
    if (result.passed) {
      std::cout << "[PASS] " << name << "\n";
    } else {
      std::cout << "[FAIL] " << name << "\n" << result.failures;
    }
    return result.passed;
  };

  bool all_passed = true;

  std::cout << "--- Triangles (static size = 3) ---\n";
  all_passed &= run_test("No duplicates", test_triangles_no_duplicates);
  all_passed &=
      run_test("Identical duplicate", test_triangles_identical_duplicate);
  all_passed &=
      run_test("Reversed duplicate", test_triangles_reversed_duplicate);
  all_passed &= run_test("Rotated duplicate", test_triangles_rotated_duplicate);
  all_passed &=
      run_test("Multiple duplicates", test_triangles_multiple_duplicates);

  std::cout << "\n--- Dynamic size polygons ---\n";
  all_passed &= run_test("Quad rotated", test_dynamic_quad_rotated);
  all_passed &= run_test("Quad reversed", test_dynamic_quad_reversed);
  all_passed &= run_test("Pentagon rotated", test_dynamic_pentagon_rotated);

  std::cout << "\n"
            << (all_passed ? "All tests passed!" : "Some tests failed!")
            << "\n";

  return all_passed ? 0 : 1;
}
