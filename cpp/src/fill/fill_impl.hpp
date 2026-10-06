/*
 * Copyright (c) 2026 XLAB
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

#include "trueform/cpp/fill/fill_holes.hpp"
#include "trueform/cpp/fill/filled_mesh.hpp"

#include "trueform/core/algorithm/generate_offset_blocks.hpp"
#include "trueform/core/algorithm/parallel_fill.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/views/indirect_range.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/fill/hole_fill_report.hpp"
#include "trueform/fill/fill_holes.hpp"
#include "trueform/fill/hole_fill_config.hpp"
#include "trueform/fill/hole_fill_result.hpp"
#include "trueform/fill/hole_split.hpp"
#include "trueform/fill/make_filled_mesh.hpp"
#include "trueform/topology/boundary_rims.hpp"
#include "trueform/topology/policy/face_membership.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace detail {

inline auto require_hole_fill_config(const tf::hole_fill_config &config)
    -> void {
  if (!std::isfinite(config.min_quality) || config.min_quality < 0 ||
      config.min_quality > 1)
    throw std::invalid_argument(
        "fill_holes: min quality must be finite and in [0, 1]");
}

template <typename Index>
auto require_rim_ids(const nd_array<Index> &rim_ids,
                     const tf::boundary_rims<Index> &rims) -> void {
  if (!rim_ids.is_valid() || rim_ids.ndim() != 1)
    throw std::invalid_argument(
        "fill_holes: rim ids must be a valid one-dimensional array");
  tf::buffer<bool> named;
  named.allocate(rims.size());
  tf::parallel_fill(named, false);
  for (const auto id : rim_ids) {
    if (id < Index{0} || static_cast<std::size_t>(id) >= rims.size())
      throw std::out_of_range("fill_holes: rim id out of range");
    const auto rim = static_cast<std::size_t>(id);
    if (!rims.closed[rim])
      throw std::invalid_argument("fill_holes: rim id names an open rim");
    if (named[rim])
      throw std::invalid_argument("fill_holes: duplicate rim id");
    named[rim] = true;
  }
}

/// The named rims as a rims carrier of their own, block `g` being
/// `rim_ids[g]`.
template <typename Index>
auto named_hole_rims(const tf::boundary_rims<Index> &rims,
                     const nd_array<Index> &rim_ids)
    -> tf::boundary_rims<Index> {
  const auto ids = rim_ids.make_range();
  const auto gather = [](const auto &block, tf::buffer<Index> &data) {
    for (const auto value : block)
      data.push_back(value);
  };
  tf::boundary_rims<Index> named;
  tf::generate_offset_blocks(tf::make_indirect_range(ids, rims.vertices),
                             named.vertices, gather, tf::checked);
  tf::generate_offset_blocks(tf::make_indirect_range(ids, rims.faces),
                             named.faces, gather, tf::checked);
  named.closed.allocate(ids.size());
  tf::parallel_fill(named.closed, true);
  return named;
}

/// A group's offending rim is its position among the rims the fill was
/// handed; the caller named those, so the report speaks the caller's ids.
template <typename Index, typename Real>
auto name_offending_rims(tf::hole_fill_result<Index, Real> &fill,
                         const nd_array<Index> &rim_ids) -> void {
  tf::parallel_for_each(
      fill.offending,
      [&rim_ids](auto &&offending) {
        if (offending[0] != Index(-1))
          offending[0] = rim_ids[static_cast<std::size_t>(offending[0])];
      },
      tf::checked);
}

template <typename T>
auto hole_fill_lane(const std::shared_ptr<void> &owner, tf::buffer<T> &values)
    -> nd_array<T> {
  return nd_array<T>::from_borrowed(owner, values.data(), values.size(),
                                    {static_cast<int>(values.size())});
}

/// A lane of fixed-width records read as its [n, Width] scalars, which the
/// packed record is.
template <typename T, std::size_t Width, typename Record>
auto hole_fill_record_lane(const std::shared_ptr<void> &owner,
                           tf::buffer<Record> &records) -> nd_array<T> {
  static_assert(sizeof(Record) == Width * sizeof(T),
                "a record lane reads its records as packed scalars");
  T *const data = records.size() ? records.begin()->data() : nullptr;
  return nd_array<T>::from_borrowed(
      owner, data, records.size() * Width,
      {static_cast<int>(records.size()), static_cast<int>(Width)});
}

template <typename Status>
auto hole_fill_status_lane(const tf::buffer<Status> &statuses)
    -> nd_array<std::int8_t> {
  tf::buffer<std::int8_t> lane;
  lane.allocate(statuses.size());
  tf::parallel_transform(
      statuses, lane,
      [](Status status) { return static_cast<std::int8_t>(status); },
      tf::checked);
  return nd_array<std::int8_t>::from_buffer(
      std::move(lane), {static_cast<int>(statuses.size())});
}

template <typename Index>
auto hole_split_lane(const tf::buffer<tf::hole_split<Index>> &splits)
    -> nd_array<Index> {
  tf::buffer<Index> lane;
  lane.allocate(splits.size() * 5);
  tf::parallel_for_each(
      tf::make_sequence_range(splits.size()),
      [&lane, &splits](std::size_t k) {
        const auto &split = splits[k];
        auto *const row = lane.begin() + 5 * k;
        row[0] = split.face;
        row[1] = split.v0;
        row[2] = split.v1;
        row[3] = split.point;
        row[4] = Index(split.parameter);
      },
      tf::checked);
  return nd_array<Index>::from_buffer(std::move(lane),
                                      {static_cast<int>(splits.size()), 5});
}

template <typename Index, typename Real>
auto make_hole_fill_report(tf::hole_fill_result<Index, Real> &&fill,
                           const mesh<Index, Real, 3> &value)
    -> hole_fill_report<Index, Real> {
  auto result =
      std::make_shared<tf::hole_fill_result<Index, Real>>(std::move(fill));
  const std::shared_ptr<void> owner = result;

  hole_fill_report<Index, Real> report;
  report.status = hole_fill_status_lane(result->status);
  report.refined = hole_fill_status_lane(result->refined);
  report.faired = hole_fill_status_lane(result->faired);
  report.seam_max_angle = hole_fill_lane(owner, result->seam_max_angle);
  report.offending = hole_fill_lane(owner, result->offending.data_buffer());
  report.offending.set_shape({static_cast<int>(result->size()), 2});
  report.group_offsets = hole_fill_lane(owner, result->group_offsets);
  report.triangle_offsets =
      hole_fill_lane(owner, result->triangles.offsets_buffer());
  report.triangles =
      hole_fill_record_lane<Index, 3>(owner, result->triangles.data_buffer());
  report.minted_point_offsets =
      hole_fill_lane(owner, result->minted_points.offsets_buffer());
  report.minted_points = hole_fill_record_lane<Real, 3>(
      owner, result->minted_points.data_buffer());
  report.splits = hole_split_lane(result->splits);
  report.plan_faces = hole_fill_lane(owner, result->plan_faces);
  report.plan_triangle_offsets =
      hole_fill_lane(owner, result->plan_triangles.offsets_buffer());
  report.plan_triangles = hole_fill_record_lane<Index, 3>(
      owner, result->plan_triangles.data_buffer());
  report.number_of_points = static_cast<Index>(value.number_of_points());
  report.number_of_faces = static_cast<Index>(value.number_of_faces());
  report.result = std::move(result);
  return report;
}

} // namespace detail

template <typename Index, typename Real>
auto fill_holes(const mesh<Index, Real, 3> &value, tf::hole_fill_config config)
    -> hole_fill_report<Index, Real> {
  detail::require_hole_fill_config(config);
  const auto membership = value.face_membership();
  const auto rims = tf::make_boundary_rims(value.faces(), membership);
  return detail::make_hole_fill_report(
      tf::fill_holes(value.polygons() | tf::tag(membership), rims, config),
      value);
}

template <typename Index, typename Real>
auto fill_holes(const mesh<Index, Real, 3> &value,
                const nd_array<Index> &rim_ids, tf::hole_fill_config config)
    -> hole_fill_report<Index, Real> {
  detail::require_hole_fill_config(config);
  const auto membership = value.face_membership();
  const auto rims = tf::make_boundary_rims(value.faces(), membership);
  detail::require_rim_ids(rim_ids, rims);
  auto fill = tf::fill_holes(value.polygons() | tf::tag(membership),
                             detail::named_hole_rims(rims, rim_ids), config);
  detail::name_offending_rims(fill, rim_ids);
  return detail::make_hole_fill_report(std::move(fill), value);
}

template <typename Index, typename Real>
auto filled_mesh(const mesh<Index, Real, 3> &value,
                 const hole_fill_report<Index, Real> &report)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  if (!report.result)
    throw std::invalid_argument("filled_mesh: the report holds no fill");
  if (value.number_of_points() !=
          static_cast<std::size_t>(report.number_of_points) ||
      value.number_of_faces() !=
          static_cast<std::size_t>(report.number_of_faces))
    throw std::invalid_argument(
        "filled_mesh: the report was stated against another mesh");
  value.require_indices();
  return tf::make_filled_mesh(value.polygons(), *report.result);
}

} // namespace tf::cpp
