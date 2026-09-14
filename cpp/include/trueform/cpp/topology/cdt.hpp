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
#pragma once

#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>

namespace tf::cpp {

template <typename Real> struct cdt_result {
  nd_array<std::int32_t> faces;
  nd_array<Real> points;
};

template <typename Real> struct cdt_result_with_map {
  nd_array<std::int32_t> faces;
  nd_array<Real> points;
  index_map<> point_index_map;
};

template <typename Real>
auto make_cdt(const nd_array<Real> &points) -> cdt_result<Real>;
template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points)
    -> cdt_result_with_map<Real>;
template <typename Real>
auto make_cdt(const nd_array<Real> &points, const nd_array<std::int32_t> &edges,
              bool split_constraints = true) -> cdt_result<Real>;
template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        bool split_constraints = true)
    -> cdt_result_with_map<Real>;
template <typename Real>
auto make_cdt(const nd_array<Real> &points, const nd_array<std::int32_t> &edges,
              const nd_array<std::int8_t> &edge_mask,
              bool split_constraints = true) -> cdt_result<Real>;
template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        const nd_array<std::int8_t> &edge_mask,
                        bool split_constraints = true)
    -> cdt_result_with_map<Real>;

#define TF_CPP_EXTERN_CDT(Real)                                                \
  extern template auto make_cdt(const nd_array<Real> &) -> cdt_result<Real>;   \
  extern template auto make_cdt_with_maps(const nd_array<Real> &)              \
      -> cdt_result_with_map<Real>;                                            \
  extern template auto make_cdt(const nd_array<Real> &,                        \
                                const nd_array<std::int32_t> &, bool)          \
      -> cdt_result<Real>;                                                     \
  extern template auto make_cdt_with_maps(const nd_array<Real> &,              \
                                          const nd_array<std::int32_t> &,      \
                                          bool) -> cdt_result_with_map<Real>;  \
  extern template auto make_cdt(                                               \
      const nd_array<Real> &, const nd_array<std::int32_t> &,                  \
      const nd_array<std::int8_t> &, bool) -> cdt_result<Real>;                \
  extern template auto make_cdt_with_maps(                                     \
      const nd_array<Real> &, const nd_array<std::int32_t> &,                  \
      const nd_array<std::int8_t> &, bool) -> cdt_result_with_map<Real>

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_EXTERN_CDT)

#undef TF_CPP_EXTERN_CDT

} // namespace tf::cpp
