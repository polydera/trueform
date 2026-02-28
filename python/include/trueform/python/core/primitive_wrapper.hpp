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
 * Author: Ziga Sajovic
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstddef>

namespace tf::py {

// ============================================================================
// Primitive type enum (mirrors TS PrimitiveType)
// ============================================================================

enum class prim_type : int {
  point = 0,
  segment = 1,
  triangle = 2,
  ray = 3,
  line = 4,
  plane = 5,
  aabb = 6,
  polygon = 7
};

// ============================================================================
// Helpers
// ============================================================================

/// Expected ndim for a single (non-batched) element.
inline auto single_ndim(prim_type t) -> int {
  switch (t) {
  case prim_type::point:
  case prim_type::plane:
    return 1;
  default:
    return 2;
  }
}

/// Number of RealT elements per single element for a given primitive type.
template <std::size_t Dims>
inline auto prim_stride(prim_type t, int n_poly_verts = 0) -> int {
  switch (t) {
  case prim_type::point:
    return static_cast<int>(Dims);
  case prim_type::segment:
    return static_cast<int>(2 * Dims);
  case prim_type::triangle:
    return static_cast<int>(3 * Dims);
  case prim_type::ray:
    return static_cast<int>(2 * Dims);
  case prim_type::line:
    return static_cast<int>(2 * Dims);
  case prim_type::plane:
    return static_cast<int>(Dims + 1);
  case prim_type::aabb:
    return static_cast<int>(2 * Dims);
  case prim_type::polygon:
    return static_cast<int>(n_poly_verts * Dims);
  }
  return 0;
}

// ============================================================================
// primitive_wrapper<Dims, RealT>
// ============================================================================

template <std::size_t Dims, typename RealT> class primitive_wrapper {
  prim_type _type;
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::c_contig> _data;

public:
  primitive_wrapper(
      int type_int,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::c_contig> data)
      : _type(static_cast<prim_type>(type_int)), _data(std::move(data)) {}

  auto type() const -> prim_type { return _type; }

  auto type_int() const -> int { return static_cast<int>(_type); }

  auto data() const
      -> const nanobind::ndarray<nanobind::numpy, RealT, nanobind::c_contig> & {
    return _data;
  }

  auto data_ptr() const -> const RealT * { return _data.data(); }

  auto ndim() const -> std::size_t { return _data.ndim(); }

  auto shape(std::size_t axis) const -> std::size_t {
    return _data.shape(axis);
  }

  auto is_batch() const -> bool {
    return static_cast<int>(ndim()) > single_ndim(_type);
  }

  auto count() const -> int {
    return is_batch() ? static_cast<int>(shape(0)) : 1;
  }

  static constexpr auto dims() -> std::size_t { return Dims; }

  auto poly_verts() const -> int {
    if (_type == prim_type::polygon)
      return static_cast<int>(shape(ndim() - 2));
    if (_type == prim_type::triangle)
      return 3;
    return 0;
  }

  auto stride() const -> int { return prim_stride<Dims>(_type, poly_verts()); }

  auto element_ptr(int i) const -> const RealT * {
    return is_batch() ? data_ptr() + i * stride() : data_ptr();
  }
};

// ============================================================================
// Registration helper
// ============================================================================

template <std::size_t Dims, typename RealT>
auto register_primitive_wrapper(nanobind::module_ &m, const char *name)
    -> void {
  namespace nb = nanobind;
  using PW = primitive_wrapper<Dims, RealT>;

  nb::class_<PW>(m, name)
      .def(nb::init<int, nb::ndarray<nb::numpy, RealT, nb::c_contig>>(),
           nb::arg("type"), nb::arg("data"))
      .def("type_int", &PW::type_int)
      .def("is_batch", &PW::is_batch)
      .def("count", &PW::count)
      .def("dims", &PW::dims)
      .def("stride", &PW::stride)
      .def("poly_verts", &PW::poly_verts)
      .def("ndim", &PW::ndim)
      .def_prop_ro("data_array",
                    [](const PW &self) { return self.data(); });
}

} // namespace tf::py
