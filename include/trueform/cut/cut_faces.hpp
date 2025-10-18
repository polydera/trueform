/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../intersect/types/simple_intersections.hpp"
#include "./loop/cut_faces.hpp"

namespace tf {
/*template <typename Index> class cut_faces : public loop::cut_faces<Index> {*/
/*  using base_t = loop::cut_faces<Index>;*/
/**/
/*public:*/
/*  template <typename Policy, typename RealT, std::size_t Dims>*/
/*  auto*/
/*  build(const tf::polygons<Policy> &_polygons,*/
/*        const tf::intersect::simple_intersections<Index, RealT, Dims> &tgs) {*/
/*    base_t::clear();*/
/*    auto polygons = tf::wrap_map(_polygons, [](auto &&x) {*/
/*      return tf::core::make_polygons(x.faces(),*/
/*                                     x.points().template as<RealT>());*/
/*    });*/
/**/
/*    base_t::build(*/
/*        tgs.intersections(), polygons,*/
/*        tf::make_points(tgs.intersection_points()),*/
/*        [&tgs](const auto &x) { return tgs.get_flat_index(x); },*/
/*        std::thread::hardware_concurrency() * 5);*/
/*  }*/
/*};*/
} // namespace tf
