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

#include "trueform/cpp/iso/isocontours.hpp"
#include "trueform/iso/make_isocontours.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace isocontours_detail {

template <typename Form, typename Real>
auto require_scalar_field(const Form &value, const nd_array<Real> &scalars)
    -> void {
  if (!scalars.is_valid() || scalars.ndim() != 1 ||
      scalars.shape_at(0) != static_cast<int>(value.number_of_points()))
    throw std::invalid_argument(
        "isocontours: scalars must have shape [number_of_points]");
}

template <typename Real>
auto require_cut_values(const nd_array<Real> &cut_values) -> void {
  if (!cut_values.is_valid() || cut_values.ndim() != 1)
    throw std::invalid_argument("isocontours: cut values must have shape [N]");
}

/// The field is stated on the operand's own points, so the mesh is read where
/// it was authored.
template <typename Index, typename Real, std::size_t Ngon, typename Cut>
auto run(const mesh<Index, Real, 3, Ngon> &value, const nd_array<Real> &scalars,
         const Cut &cut) -> tf::curves_buffer<Index, Real, 3> {
  return tf::make_isocontours(value.at_identity().topology_form(),
                              scalars.make_range(), cut);
}

template <typename Index, typename Real, std::size_t Ngon>
auto isocontours_impl(const mesh<Index, Real, 3, Ngon> &value,
                      const nd_array<Real> &scalars, Real cut_value)
    -> tf::curves_buffer<Index, Real, 3> {
  require_scalar_field(value, scalars);
  value.require_indices();
  if (value.number_of_faces() == 0)
    return {};
  return run(value, scalars, cut_value);
}

template <typename Index, typename Real, std::size_t Ngon>
auto isocontours_impl(const mesh<Index, Real, 3, Ngon> &value,
                      const nd_array<Real> &scalars,
                      const nd_array<Real> &cut_values)
    -> tf::curves_buffer<Index, Real, 3> {
  require_scalar_field(value, scalars);
  require_cut_values(cut_values);
  value.require_indices();
  if (value.number_of_faces() == 0 || cut_values.empty())
    return {};
  return run(value, scalars, cut_values.make_range());
}

} // namespace isocontours_detail

template <typename Index, typename Real, std::size_t Ngon>
auto isocontours(const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars, Real cut_value)
    -> tf::curves_buffer<Index, Real, 3> {
  return isocontours_detail::isocontours_impl(value, scalars, cut_value);
}

template <typename Index, typename Real, std::size_t Ngon>
auto isocontours(const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars,
                 const nd_array<Real> &cut_values)
    -> tf::curves_buffer<Index, Real, 3> {
  return isocontours_detail::isocontours_impl(value, scalars, cut_values);
}

} // namespace tf::cpp
