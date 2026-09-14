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
#include "clean_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_TYPED_CLEAN_SOUP(Real, Index, Dims)                 \
  template auto cleaned_polygon_soup<Index, Real, Dims, 2>(                    \
      const nd_array<Real> &, Real, bool, bool)                                \
      -> basic_cleaned_polygon_soup_result<Index, Real, Dims, 2>;              \
  template auto cleaned_polygon_soup<Index, Real, Dims, 3>(                    \
      const nd_array<Real> &, Real, bool, bool)                                \
      -> basic_cleaned_polygon_soup_result<Index, Real, Dims, 3>

TF_CPP_INSTANTIATE_TYPED_CLEAN_SOUP(double, std::int64_t, 2);

#undef TF_CPP_INSTANTIATE_TYPED_CLEAN_SOUP

} // namespace tf::cpp
