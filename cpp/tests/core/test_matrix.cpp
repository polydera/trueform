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
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/point_cloud_cache.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace {

// The matrix speaks only about the types the layer has kernels for. A
// coordinate type it has none for is refused where it always was — by the
// entry — so a detection idiom sees a substitution failure and not a hard
// error at the carrier.
static_assert(tf::cpp::matrix_knows_real_v<float>);
static_assert(tf::cpp::matrix_knows_real_v<double>);
static_assert(!tf::cpp::matrix_knows_real_v<long double>);
static_assert(tf::cpp::matrix_carries_real_v<long double>);
static_assert(tf::cpp::matrix_carries_real_v<int>);

// A carrier of a real the layer does not know can still be NAMED, which is
// what the entries' refusal probes ask of it.
static_assert(sizeof(tf::cpp::cache<std::int32_t, long double, 3, 3>) > 0);
static_assert(sizeof(tf::cpp::point_cloud_cache<long double, 3>) > 0);

// What this build carries, stated by the same header the archive was
// generated from.
static_assert(tf::cpp::matrix_carries_v<tf::cpp::default_index_t, float, 3> ==
              (TF_CPP_MATRIX_HAS_INT32 && TF_CPP_MATRIX_HAS_FLOAT &&
               TF_CPP_MATRIX_HAS_3D));
static_assert(tf::cpp::matrix_carries_points_v<double, 2> ==
              (TF_CPP_MATRIX_HAS_DOUBLE && TF_CPP_MATRIX_HAS_2D));

} // namespace

TEST_CASE("the built matrix states itself to the compiler",
          "[cpp][core][matrix]") {
  CHECK(tf::cpp::matrix_has_real_v<float> == bool{TF_CPP_MATRIX_HAS_FLOAT});
  CHECK(tf::cpp::matrix_has_real_v<double> == bool{TF_CPP_MATRIX_HAS_DOUBLE});
  CHECK(tf::cpp::matrix_has_index_v<std::int32_t> ==
        bool{TF_CPP_MATRIX_HAS_INT32});
  CHECK(tf::cpp::matrix_has_index_v<std::int64_t> ==
        bool{TF_CPP_MATRIX_HAS_INT64});
  CHECK(tf::cpp::matrix_has_dims_v<2> == bool{TF_CPP_MATRIX_HAS_2D});
  CHECK(tf::cpp::matrix_has_dims_v<3> == bool{TF_CPP_MATRIX_HAS_3D});
}
