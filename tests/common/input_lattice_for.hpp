/**
 * @file input_lattice_for.hpp
 * @brief The operands' lattice view, as the pipeline's factory states it
 *
 * Every build below @ref tf::arrangement_graph is handed the view rather than
 * making one, so a suite that drives @ref tf::polygon_intersections directly
 * states it here, at the same arity the factory would.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/core/coordinate_type.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/vertex_converter.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/exact/input_lattice.hpp>

#include <type_traits>

namespace tf::test {

template <typename Int = tf::none_t, typename Policy>
auto input_lattice_for(const tf::polygons<Policy> &form, double tolerance) {
  using index_t = std::decay_t<decltype(form.faces()[0][0])>;
  using real_t = tf::coordinate_type<Policy>;
  using int_t = tf::exact::resolve_int_type<Int, real_t>;
  tf::exact::input_lattice<index_t, real_t, int_t> lattice;
  lattice.build(tf::exact::make_vertex_converter<int_t, real_t>(form),
                [&form](index_t, auto &&f) { f(form); }, index_t(1), tolerance);
  return lattice;
}

template <typename Int = tf::none_t, typename Policy0, typename Policy1>
auto input_lattice_for(const tf::polygons<Policy0> &form0,
                       const tf::polygons<Policy1> &form1, double tolerance) {
  using index_t = std::decay_t<decltype(form0.faces()[0][0])>;
  using real_t = tf::coordinate_type<Policy0, Policy1>;
  using int_t = tf::exact::resolve_int_type<Int, real_t>;
  tf::exact::input_lattice<index_t, real_t, int_t> lattice;
  lattice.build(tf::exact::make_vertex_converter<int_t, real_t>(form0, form1),
                [&form0, &form1](index_t tag, auto &&f) {
                  if (tag == index_t(0))
                    f(form0);
                  else
                    f(form1);
                },
                index_t(2), tolerance);
  return lattice;
}

template <typename Int = tf::none_t, typename Iterator, std::size_t N>
auto input_lattice_for(tf::range<Iterator, N> forms, double tolerance) {
  using form_t = std::decay_t<decltype(forms[0])>;
  using index_t = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using real_t = tf::coordinate_type<form_t>;
  using int_t = tf::exact::resolve_int_type<Int, real_t>;
  tf::exact::input_lattice<index_t, real_t, int_t> lattice;
  lattice.build(tf::exact::make_vertex_converter<int_t, real_t>(forms),
                [forms](index_t tag, auto &&f) { f(forms[tag]); },
                index_t(forms.size()), tolerance);
  return lattice;
}

} // namespace tf::test
