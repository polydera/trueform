/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../clean/soup/segments.hpp"
#include "../topology/planar_embedding.hpp"
#include "./intersected_segments.hpp"
namespace tf {
template <typename Index, typename RealType>
class planar_arrangements
    : public tf::planar_embedding<Index, RealType>,
      public tf::intersected_segments<Index, RealType, 2> {
  using pe_base_t = tf::planar_embedding<Index, RealType>;
  using is_base_t = tf::intersected_segments<Index, RealType, 2>;

public:
  template <typename Policy> auto build(const tf::segments<Policy> &segments) {
    clear();
    _cs.build(segments);
    _em.build(_cs.segments());
    _tree.build(_cs.segments(), tf::config_tree(4, 4));
    _si.build(_cs.segments() | tf::tag(_em), _tree);
    is_base_t::build(_cs.segments(), _si);
    _work_buffer.allocate(is_base_t::edges().size() * 4);
    tf::parallel_apply(
        tf::zip(is_base_t::edges(), tf::make_blocked_range<4>(_work_buffer)),
        [](auto pair) {
          auto &&[_in, _out] = pair;
          _out[0] = _in[0];
          _out[1] = _in[1];
          _out[2] = _in[1];
          _out[3] = _in[0];
        });
    pe_base_t::build(tf::make_edges(tf::make_blocked_range<2>(_work_buffer)),
                     is_base_t::points());
  }

  auto clear() {
    _em.clear();
    _tree.clear();
    _cs.clear();
    _si.clear();
    is_base_t::clear();
    pe_base_t::clear();
  }

private:
  tf::edge_membership<Index> _em;
  tf::tree<Index, RealType, 2> _tree;
  tf::clean::segment_soup<Index, RealType, 2> _cs;
  tf::segment_intersections<Index, RealType, 2> _si;
  tf::buffer<Index> _work_buffer;
};
} // namespace tf
