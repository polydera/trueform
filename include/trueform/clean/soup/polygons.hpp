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
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/remove_if_and_make_map.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../topology/compute_unique_faces_mask.hpp"
#include "../../topology/face_membership.hpp"
#include "../index_map/points.hpp"

namespace tf::clean {
template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngons>
class polygon_soup : public polygons_buffer<Index, RealT, Dims, Ngons> {
  using base_t = polygons_buffer<Index, RealT, Dims, Ngons>;

public:
  template <typename Policy>
  auto build(const tf::polygons<Policy> &polygons, RealT tolerance = 0) {
    clear();
    make_initial_points(polygons, tolerance);
    make_faces();
    remove_uncontained_points();
  }

  auto build(tf::buffer<RealT> &&polygons, RealT tolerance = 0) {
    clear();
    make_initial_points(std::move(polygons), tolerance);
    make_faces();
    remove_uncontained_points();
  }

  template <typename Policy>
  auto build_and_deduplicate_faces(const tf::polygons<Policy> &polygons,
                                   RealT tolerance = 0) {
    build(polygons, tolerance);
    remove_duplicate_faces();
  }

  auto build_and_deduplicate_faces(tf::buffer<RealT> &&polygons,
                                   RealT tolerance = 0) {
    build(std::move(polygons), tolerance);
    remove_duplicate_faces();
  }

  auto clear() {
    base_t::clear();
    _im.f().clear();
    _im.kept_ids().clear();
    _mask.clear();
  }

private:
  template <typename Policy>
  auto make_initial_points(const tf::polygons<Policy> &polygons,
                           RealT tolerance) {
    base_t::points_buffer().allocate(polygons.size() * Ngons);
    auto points = base_t::points();
    tf::parallel_for_each(tf::zip(polygons, tf::make_blocked_range<Ngons>(points)),
                       [](auto pair) {
                         auto &&[_in, _out] = pair;
                         std::copy(_in.begin(), _in.end(), _out.begin());
                       });
    make_initial_points_impl(tolerance);
  }

  auto make_initial_points(tf::buffer<RealT> &&polygons, RealT tolerance) {
    base_t::points_buffer() = std::move(polygons);
    make_initial_points_impl(tolerance);
  }

  auto make_initial_points_impl(RealT tolerance) {
    if (tolerance <= 0)
      return make_initial_points_impl();
    auto points = base_t::points();
    tf::make_clean_index_map(points, tolerance, _im);
    tf::points_buffer<RealT, Dims> tmp_buffer;
    tmp_buffer.allocate(_im.kept_ids().size());
    tf::parallel_copy(tf::make_indirect_range(_im.kept_ids(), points),
                      tmp_buffer);
    base_t::points_buffer() = std::move(tmp_buffer);
  }

  auto make_initial_points_impl() {
    auto points = base_t::points();
    tf::make_unique_and_index_map(points, _im);
    base_t::points_buffer().data_buffer().erase_till_end(
        base_t::points_buffer().data_buffer().begin() +
        _im.kept_ids().size() * Dims);
  }

  auto make_faces() {
    _mask.allocate(_im.kept_ids().size());
    tf::parallel_fill(_mask, false);
    tf::parallel_for(tf::make_blocked_range<Ngons>(_im.f()), [&](auto begin,
                                                                 auto end) {
      for (auto face : tf::make_range(begin, end)) {
        if constexpr (Ngons == 3) {
          if (face[0] == face[1] || face[0] == face[2] || face[1] == face[2])
            for (auto &e : face)
              e = -1;
          else
            for (auto &e : face)
              _mask[e] = true;

        } else {
          std::array<Index, Ngons> array;
          std::copy(face.begin(), face.end(), array.begin());
          std::sort(array.begin(), array.end());
          if (std::unique(array.begin(), array.end()) - array.begin() <= 2)
            for (auto &e : face)
              e = -1;
          else
            for (auto &e : face)
              _mask[e] = true;
        }
      }
    });
    base_t::faces_buffer().data_buffer().allocate(_im.f().size());
    base_t::faces_buffer().data_buffer().erase_till_end(
        std::copy_if(_im.f().begin(), _im.f().end(),
                     base_t::faces_buffer().data_buffer().begin(),
                     [](auto x) { return x != -1; }));
  }

  auto remove_uncontained_points() {
    auto &map = _im.f();
    map.allocate(base_t::points().size());
    auto r = tf::zip(_mask, base_t::points());
    auto n_kept = tf::remove_if_and_make_map(
                      r, [](auto &&pair) { return !get<0>(pair); }, map) -
                  r.begin();
    base_t::points_buffer().erase(base_t::points_buffer().begin() + n_kept,
                                  base_t::points_buffer().end());
    tf::parallel_for_each(
        base_t::faces_buffer(),
        [&](auto &&face) {
          for (auto &e : face)
            e = map[e];
        },
        tf::checked);
  }

  auto remove_duplicate_faces() {
    if (base_t::faces().size() < 2)
      return;
    tf::face_membership<Index> fm;
    fm.build(base_t::polygons());
    _mask.allocate(base_t::faces().size());
    tf::compute_unique_faces_mask(base_t::faces(), fm, _mask);
    auto zipped = tf::zip(base_t::faces_buffer(), _mask);
    auto new_end = std::remove_if(zipped.begin(), zipped.end(),
                                  [](auto &&pair) { return !get<1>(pair); });
    auto n_kept = new_end - zipped.begin();
    base_t::faces_buffer().erase(base_t::faces_buffer().begin() + n_kept,
                                 base_t::faces_buffer().end());
  }

  tf::index_map_buffer<Index> _im;
  tf::buffer<bool> _mask;
};
// ---------------------------------------------------------------------------
// Dynamic polygon soup specialization (variable-sized polygons)
// ---------------------------------------------------------------------------
template <typename Index, typename RealT, std::size_t Dims>
class polygon_soup<Index, RealT, Dims, tf::dynamic_size>
    : public polygons_buffer<Index, RealT, Dims, tf::dynamic_size> {
  using base_t = polygons_buffer<Index, RealT, Dims, tf::dynamic_size>;

public:
  template <typename Policy>
  auto build(const tf::polygons<Policy> &polygons, RealT tolerance = 0) {
    clear();
    make_initial_points(polygons, tolerance);
    make_faces();
    remove_uncontained_points();
  }

  template <typename Policy>
  auto build_and_deduplicate_faces(const tf::polygons<Policy> &polygons,
                                   RealT tolerance = 0) {
    build(polygons, tolerance);
    remove_duplicate_faces();
  }

  auto clear() {
    base_t::clear();
    _im.f().clear();
    _im.kept_ids().clear();
    _mask.clear();
    _offsets_buf.clear();
  }

private:
  template <typename Policy>
  auto make_initial_points(const tf::polygons<Policy> &polygons,
                           RealT tolerance) {
    // Build offsets from per-polygon sizes
    _offsets_buf.allocate(polygons.size() + 1);
    _offsets_buf[0] = 0;
    std::size_t total = 0;
    for (auto [i, poly] : tf::enumerate(polygons)) {
      total += poly.size();
      _offsets_buf[i + 1] = static_cast<Index>(total);
    }
    // Copy vertices into flat points buffer
    base_t::points_buffer().allocate(total);
    tf::parallel_for_each(
        tf::zip(polygons,
                tf::make_offset_block_range(_offsets_buf, base_t::points())),
        [](auto pair) {
          auto &&[in, out] = pair;
          std::copy(in.begin(), in.end(), out.begin());
        });
    make_initial_points_impl(tolerance);
  }

  auto make_initial_points_impl(RealT tolerance) {
    if (tolerance <= 0)
      return make_initial_points_impl();
    auto points = base_t::points();
    tf::make_clean_index_map(points, tolerance, _im);
    tf::points_buffer<RealT, Dims> tmp_buffer;
    tmp_buffer.allocate(_im.kept_ids().size());
    tf::parallel_copy(tf::make_indirect_range(_im.kept_ids(), points),
                      tmp_buffer);
    base_t::points_buffer() = std::move(tmp_buffer);
  }

  auto make_initial_points_impl() {
    auto points = base_t::points();
    tf::make_unique_and_index_map(points, _im);
    base_t::points_buffer().data_buffer().erase_till_end(
        base_t::points_buffer().data_buffer().begin() +
        _im.kept_ids().size() * Dims);
  }

  auto make_faces() {
    _mask.allocate(_im.kept_ids().size());
    tf::parallel_fill(_mask, false);
    // Block remapped indices using precomputed offsets
    auto obr = tf::make_offset_block_range(_offsets_buf, _im.f());
    // Mark degenerate faces (-1) and used points
    tf::parallel_for_each(
        obr,
        [&](auto face, tf::small_vector<Index, 8> &sorted) {
          sorted.clear();
          std::copy(face.begin(), face.end(), std::back_inserter(sorted));
          std::sort(sorted.begin(), sorted.end());
          if (std::unique(sorted.begin(), sorted.end()) - sorted.begin() <= 2)
            for (auto &e : face)
              e = -1;
          else
            for (auto &e : face)
              _mask[e] = true;
        },
        tf::small_vector<Index, 8>{}, tf::checked);
    // Count valid faces and total indices (parallel reduce)
    auto [n_valid, total_idx] = tf::reduce(
        tf::make_mapped_range(obr, [](auto face) {
          if (face[0] != -1)
            return std::pair<std::size_t, std::size_t>{1, face.size()};
          return std::pair<std::size_t, std::size_t>{0, 0};
        }),
        [](auto acc, auto x) {
          return std::pair{acc.first + x.first, acc.second + x.second};
        },
        std::pair<std::size_t, std::size_t>{0, 0});
    // Pre-allocate and fill directly
    auto &ob = base_t::faces_buffer();
    if (n_valid == 0)
      return;
    ob.offsets_buffer().allocate(n_valid + 1);
    ob.data_buffer().allocate(total_idx);
    auto *off = ob.offsets_buffer().data();
    auto *dat = ob.data_buffer().data();
    *off++ = 0;
    for (auto face : obr) {
      if (face[0] != -1) {
        dat = std::copy(face.begin(), face.end(), dat);
        *off++ = static_cast<Index>(dat - ob.data_buffer().data());
      }
    }
  }

  auto remove_uncontained_points() {
    auto &map = _im.f();
    map.allocate(base_t::points().size());
    auto r = tf::zip(_mask, base_t::points());
    auto n_kept = tf::remove_if_and_make_map(
                      r, [](auto &&pair) { return !get<0>(pair); }, map) -
                  r.begin();
    base_t::points_buffer().erase(base_t::points_buffer().begin() + n_kept,
                                  base_t::points_buffer().end());
    // Remap face indices in the flat data buffer
    tf::parallel_for_each(
        base_t::faces_buffer().data_buffer(),
        [&](auto &e) { e = map[e]; });
  }

  auto remove_duplicate_faces() {
    if (base_t::faces().size() < 2)
      return;
    tf::face_membership<Index> fm;
    fm.build(base_t::polygons());
    _mask.allocate(base_t::faces().size());
    tf::compute_unique_faces_mask(base_t::faces(), fm, _mask);
    // Rebuild offset_block_buffer with only unique faces
    tf::offset_block_buffer<Index, Index> kept;
    for (auto [face, keep] : tf::zip(base_t::faces_buffer(), _mask)) {
      if (keep)
        kept.push_back(face);
    }
    base_t::faces_buffer() = std::move(kept);
  }

  tf::buffer<Index> _offsets_buf;
  tf::index_map_buffer<Index> _im;
  tf::buffer<bool> _mask;
};

} // namespace tf::clean
