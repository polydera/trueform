/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./buffer.hpp"
#include "./edges.hpp"
#include "./points.hpp"
#include "./segments.hpp"
#include "./views/blocked_range.hpp"
namespace tf {

template <typename Index, typename RealT, std::size_t Dims>
class segments_buffer {
public:
  auto points() const { return tf::make_points<Dims>(_raw_points); }

  auto points() { return tf::make_points<Dims>(_raw_points); }

  auto edges() const {
    return tf::make_edges(tf::make_blocked_range<2>(_raw_edges));
  }

  auto edges() { return tf::make_edges(tf::make_blocked_range<2>(_raw_edges)); }

  auto segments() const { return tf::make_segments(edges(), points()); }

  auto segments() { return tf::make_segments(edges(), points()); }

  auto raw_edges_buffer() -> tf::buffer<Index> & { return _raw_edges; }

  auto raw_edges_buffer() const -> const tf::buffer<Index> & {
    return _raw_edges;
  }

  auto raw_points_buffer() -> tf::buffer<RealT> & { return _raw_points; }

  auto raw_points_buffer() const -> const tf::buffer<RealT> & {
    return _raw_points;
  }

  auto clear() {
    _raw_points.clear();
    _raw_edges.clear();
  }

private:
  tf::buffer<Index> _raw_edges;
  tf::buffer<RealT> _raw_points;
};
} // namespace tf
