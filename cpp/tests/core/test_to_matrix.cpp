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
#include "trueform/cpp/core/to_matrix.hpp"

#include "trueform/core/transformation.hpp"
#include "trueform/core/transformation_view.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

namespace {

template <typename Real>
auto check_values(const tf::cpp::nd_array<Real> &matrix, const Real *expected)
    -> void {
  for (std::size_t index = 0; index < matrix.length(); ++index)
    CHECK(matrix[index] == expected[index]);
}

} // namespace

TEMPLATE_TEST_CASE("to_matrix materializes a full typed affine matrix",
                   "[cpp][core][to-matrix]", float, double) {
  tf::transformation<TestType, 3> transformation{
      TestType{1}, TestType{2},  TestType{3},  TestType{4},
      TestType{5}, TestType{6},  TestType{7},  TestType{8},
      TestType{9}, TestType{10}, TestType{11}, TestType{12},
  };

  auto matrix = tf::cpp::to_matrix<TestType>(transformation);
  static_assert(std::is_same_v<decltype(matrix), tf::cpp::nd_array<TestType>>);
  CHECK((matrix.raw_shape() == tf::small_vector<int, 3>{4, 4}));

  const TestType expected[]{
      TestType{1}, TestType{2},  TestType{3},  TestType{4},
      TestType{5}, TestType{6},  TestType{7},  TestType{8},
      TestType{9}, TestType{10}, TestType{11}, TestType{12},
      TestType{0}, TestType{0},  TestType{0},  TestType{1},
  };
  check_values(matrix, expected);
}

TEST_CASE("to_matrix supports transformation policies and dtype conversion",
          "[cpp][core][to-matrix]") {
  double values[]{1.25, 2.5, 3.75, 4.0, 5.5, 6.25};
  const auto transformation = tf::make_transformation_view<2>(values);

  auto matrix = tf::cpp::to_matrix<float>(transformation);
  static_assert(std::is_same_v<decltype(matrix), tf::cpp::nd_array<float>>);
  CHECK((matrix.raw_shape() == tf::small_vector<int, 3>{3, 3}));

  const float expected[]{1.25F, 2.5F, 3.75F, 4.0F, 5.5F,
                         6.25F, 0.0F, 0.0F,  1.0F};
  check_values(matrix, expected);
}

TEST_CASE("to_matrix materializes identity transformation policies",
          "[cpp][core][to-matrix]") {
  const tf::identity_transformation<double, 3> transformation;
  const auto matrix = tf::cpp::to_matrix<double>(transformation);

  CHECK((matrix.raw_shape() == tf::small_vector<int, 3>{4, 4}));
  const double expected[]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  check_values(matrix, expected);
}
