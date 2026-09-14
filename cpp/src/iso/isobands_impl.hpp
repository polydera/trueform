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

#include "trueform/cpp/iso/isobands.hpp"

#include "trueform/iso/make_isobands.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace isobands_detail {

template <typename Form, typename Real>
auto require_inputs(const Form &value, const nd_array<Real> &scalars,
                    const nd_array<Real> &cut_values) -> void {
  if (!scalars.is_valid() || scalars.ndim() != 1 ||
      scalars.shape_at(0) != static_cast<int>(value.number_of_points()))
    throw std::invalid_argument(
        "isobands: scalars must have shape [number_of_points]");
  if (!cut_values.is_valid() || cut_values.ndim() != 1)
    throw std::invalid_argument("isobands: cut values must have shape [N]");

  for (const auto scalar : scalars)
    if (!std::isfinite(scalar))
      throw std::invalid_argument("isobands: scalars must be finite");
  for (const auto cut_value : cut_values)
    if (!std::isfinite(cut_value))
      throw std::invalid_argument("isobands: cut values must be finite");
  for (const auto point : value.points())
    for (const auto coordinate : point)
      if (!std::isfinite(coordinate))
        throw std::invalid_argument(
            "isobands: mesh coordinates must be finite");

  value.require_indices();
}

template <typename Real>
auto unique_cut_count(const nd_array<Real> &cut_values) -> std::size_t {
  if (cut_values.empty())
    return 0;
  auto sorted = cut_values.deep_copy();
  std::sort(sorted.begin(), sorted.end());
  return static_cast<std::size_t>(std::unique(sorted.begin(), sorted.end()) -
                                  sorted.begin());
}

template <typename Real>
auto require_selected_bands(const nd_array<Real> &cut_values,
                            const nd_array<std::int32_t> &selected_bands)
    -> void {
  if (!selected_bands.is_valid() || selected_bands.ndim() != 1)
    throw std::invalid_argument("isobands: selected bands must have shape [N]");

  const auto band_count = unique_cut_count(cut_values);
  for (const auto band : selected_bands)
    if (band < 0 || static_cast<std::size_t>(band) > band_count)
      throw std::out_of_range("isobands: selected band index out of range");
}

template <typename Index, typename Real, std::size_t Ngon, typename Polygons>
auto convert_result(Polygons &&polygons, tf::buffer<Index> labels,
                    tf::buffer<Index> face_labels)
    -> isobands_result<Index, Real, Ngon> {
  const auto count = static_cast<int>(labels.size());
  return {std::forward<Polygons>(polygons),
          nd_array<Index>::from_buffer(std::move(labels), {count}),
          nd_array<Index>::from_buffer(std::move(face_labels), {count})};
}

template <typename Index, typename Real, std::size_t Ngon, typename Polygons>
auto convert_result(Polygons &&polygons, tf::buffer<Index> labels,
                    tf::buffer<Index> face_labels,
                    tf::curves_buffer<Index, Real, 3> curves_buffer)
    -> isobands_with_curves_result<Index, Real, Ngon> {
  const auto count = static_cast<int>(labels.size());
  return {std::forward<Polygons>(polygons),
          nd_array<Index>::from_buffer(std::move(labels), {count}),
          nd_array<Index>::from_buffer(std::move(face_labels), {count}),
          std::move(curves_buffer)};
}

inline auto all_band_ids(std::size_t cut_count) -> tf::buffer<std::int32_t> {
  if (cut_count >=
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::length_error("isobands: too many cut values");
  tf::buffer<std::int32_t> selected_bands;
  selected_bands.allocate(cut_count + 1);
  for (std::size_t band = 0; band <= cut_count; ++band)
    selected_bands[band] = static_cast<std::int32_t>(band);
  return selected_bands;
}

template <typename Index, typename Real, std::size_t Ngon>
auto run_all(const mesh<Index, Real, 3, Ngon> &value,
             const nd_array<Real> &scalars, const nd_array<Real> &cut_values)
    -> isobands_result<Index, Real, Ngon> {
  require_inputs(value, scalars, cut_values);
  const auto selected_bands = all_band_ids(unique_cut_count(cut_values));
  auto [output, labels, face_labels] = tf::make_isobands<Index>(
      value.polygons(), scalars.make_range(), cut_values.make_range(),
      tf::make_range(selected_bands));
  return convert_result<Index, Real, Ngon>(std::move(output), std::move(labels),
                                           std::move(face_labels));
}

template <typename Index, typename Real, std::size_t Ngon>
auto run_all_with_curves(const mesh<Index, Real, 3, Ngon> &value,
                         const nd_array<Real> &scalars,
                         const nd_array<Real> &cut_values)
    -> isobands_with_curves_result<Index, Real, Ngon> {
  require_inputs(value, scalars, cut_values);
  const auto selected_bands = all_band_ids(unique_cut_count(cut_values));
  auto [output, labels, face_labels, curves] = tf::make_isobands<Index>(
      value.polygons(), scalars.make_range(), cut_values.make_range(),
      tf::make_range(selected_bands), tf::return_curves);
  return convert_result<Index, Real, Ngon>(std::move(output), std::move(labels),
                                           std::move(face_labels),
                                           std::move(curves));
}

template <typename Index, typename Real, std::size_t Ngon>
auto run_selected(const mesh<Index, Real, 3, Ngon> &value,
                  const nd_array<Real> &scalars,
                  const nd_array<Real> &cut_values,
                  const nd_array<std::int32_t> &selected_bands)
    -> isobands_result<Index, Real, Ngon> {
  require_inputs(value, scalars, cut_values);
  require_selected_bands(cut_values, selected_bands);
  auto [output, labels, face_labels] = tf::make_isobands<Index>(
      value.polygons(), scalars.make_range(), cut_values.make_range(),
      selected_bands.make_range());
  return convert_result<Index, Real, Ngon>(std::move(output), std::move(labels),
                                           std::move(face_labels));
}

template <typename Index, typename Real, std::size_t Ngon>
auto run_selected_with_curves(const mesh<Index, Real, 3, Ngon> &value,
                              const nd_array<Real> &scalars,
                              const nd_array<Real> &cut_values,
                              const nd_array<std::int32_t> &selected_bands)
    -> isobands_with_curves_result<Index, Real, Ngon> {
  require_inputs(value, scalars, cut_values);
  require_selected_bands(cut_values, selected_bands);
  auto [output, labels, face_labels, curves] = tf::make_isobands<Index>(
      value.polygons(), scalars.make_range(), cut_values.make_range(),
      selected_bands.make_range(), tf::return_curves);
  return convert_result<Index, Real, Ngon>(std::move(output), std::move(labels),
                                           std::move(face_labels),
                                           std::move(curves));
}

} // namespace isobands_detail

template <typename Index, typename Real, std::size_t Ngon>
auto isobands(const mesh<Index, Real, 3, Ngon> &value,
              const nd_array<Real> &scalars, const nd_array<Real> &cut_values)
    -> isobands_result<Index, Real, Ngon> {
  return isobands_detail::run_all(value, scalars, cut_values);
}

template <typename Index, typename Real, std::size_t Ngon>
auto isobands_with_curves(const mesh<Index, Real, 3, Ngon> &value,
                          const nd_array<Real> &scalars,
                          const nd_array<Real> &cut_values)
    -> isobands_with_curves_result<Index, Real, Ngon> {
  return isobands_detail::run_all_with_curves(value, scalars, cut_values);
}

template <typename Index, typename Real, std::size_t Ngon>
auto isobands_selected(const mesh<Index, Real, 3, Ngon> &value,
                       const nd_array<Real> &scalars,
                       const nd_array<Real> &cut_values,
                       const nd_array<std::int32_t> &selected_bands)
    -> isobands_result<Index, Real, Ngon> {
  return isobands_detail::run_selected(value, scalars, cut_values,
                                       selected_bands);
}

template <typename Index, typename Real, std::size_t Ngon>
auto isobands_with_curves_selected(const mesh<Index, Real, 3, Ngon> &value,
                                   const nd_array<Real> &scalars,
                                   const nd_array<Real> &cut_values,
                                   const nd_array<std::int32_t> &selected_bands)
    -> isobands_with_curves_result<Index, Real, Ngon> {
  return isobands_detail::run_selected_with_curves(value, scalars, cut_values,
                                                   selected_bands);
}

} // namespace tf::cpp
