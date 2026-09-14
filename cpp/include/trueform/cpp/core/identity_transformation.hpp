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

#include "trueform/core/transformation_view.hpp"
#include "trueform/cpp/core/detail/identity_transformation_storage.hpp"

#include <cstddef>

namespace tf::cpp {

template <typename Real, std::size_t Dims>
auto identity_transformation_view()
    -> tf::transformation_view<const Real, Dims> {
  return tf::make_transformation_view<Dims>(
      detail::identity_transformation_storage<Real, Dims>.data());
}

} // namespace tf::cpp
