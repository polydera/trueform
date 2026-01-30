/*
 * Boolean topology test - verifies mesh remains closed and manifold
 * after repeated boolean ops (tests coplanarity handling)
 */
#include <trueform/trueform.hpp>
#include <tbb/task_arena.h>
#include <iostream>

struct test_result {
  bool passed = true;
  std::string failures;

  auto fail(const std::string &msg) -> void {
    passed = false;
    failures += "  " + msg + "\n";
  }
};

// Test repeated boolean ops at same location (coplanarity stress test)
template <typename RealType>
auto test_repeated_boolean(tf::boolean_op op, const char *op_name,
                           int num_iterations) -> test_result {
  test_result result;

  // Create base sphere
  auto big_sphere = tf::make_sphere_mesh(RealType(10.0), 40, 40);
  tf::ensure_positive_orientation(big_sphere.polygons());

  // Create small sphere for operation
  auto small_sphere = tf::make_sphere_mesh(RealType(0.5), 20, 20);
  tf::ensure_positive_orientation(small_sphere.polygons());

  // Place small sphere at north pole of big sphere
  auto merge_point = big_sphere.points()[0];
  auto transform =
      tf::make_transformation_from_translation(merge_point.as_vector());
  auto frame = tf::make_frame(transform);

  // First boolean - establish baseline
  auto [current, labels] = tf::make_boolean(
      big_sphere.polygons(), small_sphere.polygons() | tf::tag(frame), op);

  auto baseline_points = current.points().size();
  auto baseline_faces = current.polygons().size();

  // Check first result
  auto boundaries = tf::make_boundary_paths(current.polygons());
  auto non_manifold = tf::make_non_manifold_edges(current.polygons());

  if (boundaries.size() != 0) {
    result.fail(std::string(op_name) + " iter 1: " +
                std::to_string(boundaries.size()) + " boundary loops");
  }
  if (non_manifold.size() != 0) {
    result.fail(std::string(op_name) + " iter 1: " +
                std::to_string(non_manifold.size()) + " non-manifold edges");
  }

  // Repeated boolean at same point (coplanarity test)
  for (int i = 2; i <= num_iterations; ++i) {
    auto [next, next_labels] = tf::make_boolean(
        current.polygons(), small_sphere.polygons() | tf::tag(frame), op);

    boundaries = tf::make_boundary_paths(next.polygons());
    non_manifold = tf::make_non_manifold_edges(next.polygons());

    if (boundaries.size() != 0) {
      result.fail(std::string(op_name) + " iter " + std::to_string(i) + ": " +
                  std::to_string(boundaries.size()) + " boundary loops");
    }
    if (non_manifold.size() != 0) {
      result.fail(std::string(op_name) + " iter " + std::to_string(i) + ": " +
                  std::to_string(non_manifold.size()) + " non-manifold edges");
    }

    // Check consistency: repeated ops should give same result
    if (next.points().size() != baseline_points) {
      result.fail(std::string(op_name) + " iter " + std::to_string(i) +
                  ": point count changed from " +
                  std::to_string(baseline_points) + " to " +
                  std::to_string(next.points().size()));
    }
    if (next.polygons().size() != baseline_faces) {
      result.fail(std::string(op_name) + " iter " + std::to_string(i) +
                  ": face count changed from " +
                  std::to_string(baseline_faces) + " to " +
                  std::to_string(next.polygons().size()));
    }

    current = std::move(next);
  }

  return result;
}

template <typename RealType>
auto run_all_boolean_tests(const char *precision_name) -> bool {
  bool all_passed = true;

  struct op_test {
    tf::boolean_op op;
    const char *name;
  };

  op_test ops[] = {
      {tf::boolean_op::merge, "merge"},
      {tf::boolean_op::intersection, "intersection"},
      {tf::boolean_op::left_difference, "left_difference"},
  };

  constexpr int num_iterations = 4;

  for (const auto &[op, name] : ops) {
    auto result = test_repeated_boolean<RealType>(op, name, num_iterations);
    if (!result.passed) {
      std::cout << "FAIL [" << precision_name << "] " << name << ":\n"
                << result.failures;
      all_passed = false;
    }
  }

  return all_passed;
}

auto main() -> int {
  bool passed = run_all_boolean_tests<double>("double");
  std::cout << "here" << std::endl;

  bool passed1 = false;
  tbb::task_arena arena(1);
  arena.execute([&] {
    passed1 = run_all_boolean_tests<float>("float");
  });

  if (passed && passed1) {
    std::cout << "All boolean topology tests passed." << std::endl;
  }

  return (passed && passed1) ? 0 : 1;
}
