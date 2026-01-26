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

#include <nanobind/nanobind.h>

namespace tf::py::impl {

auto make_empty_capsule() -> nanobind::capsule;

} // namespace tf::py::impl

namespace tf::py {

/**
 * Create a capsule from typed pointer with proper ownership transfer
 * Handles empty arrays safely by using a shared dummy allocation
 * @param data Pointer to data (can be nullptr for empty arrays)
 * @return Capsule with appropriate deleter
 */
template <typename T> auto make_capsule(T *data) -> nanobind::capsule {
  if (!data)
    return impl::make_empty_capsule();
  else
    return nanobind::capsule(
        data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
}

} // namespace tf::py
