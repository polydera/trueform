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

#include <memory>
#include <trueform/python/util/make_capsule.hpp>

namespace tf::py::impl {

namespace {
// Global dummy allocation for empty numpy arrays
// nanobind crashes if we pass nullptr as data pointer, even for size-0 arrays
// We use a single shared allocation for all empty arrays
std::unique_ptr<char> empty_dummy{new char[1]};
} // anonymous namespace

auto make_empty_capsule() -> nanobind::capsule {
  // Empty array - use shared dummy allocation with no-op deleter
  return nanobind::capsule(empty_dummy.get(), [](void *) noexcept {});
}

} // namespace tf::py::impl
