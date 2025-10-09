/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/parallel_apply.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/remove_if_and_make_map.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/zip.hpp"
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
    tf::parallel_apply(tf::zip(polygons, tf::make_blocked_range<Ngons>(points)),
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
    tf::parallel_apply(
        base_t::faces_buffer(),
        [&](auto &&face) {
          for (auto &e : face)
            e = map[e];
        },
        tf::checked);
  }

  tf::index_map_buffer<Index> _im;
  tf::buffer<bool> _mask;
};
} // namespace tf::clean
