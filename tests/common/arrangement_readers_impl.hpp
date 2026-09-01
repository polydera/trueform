/**
 * @file arrangement_readers_impl.hpp
 * @brief The arrangement read tier's bodies
 *
 * Included ONLY by a combination's builder translation unit.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "arrangement_readers.hpp"

#include <trueform/arrangement/make_arrangement_mesh.hpp>
#include <trueform/arrangement/make_mesh_arrangements.hpp>
#include <trueform/arrangement/make_polygon_arrangements.hpp>
#include <trueform/arrangement/make_intersection_curves.hpp>
#include <trueform/csg/make_intersection_curves.hpp>
#include <trueform/core/none.hpp>

#include <utility>

namespace tf::test {

template <typename Graph>
auto arrangement_mesh_of(const Graph &graph) -> arrangement_mesh_t<Graph> {
  return tf::make_arrangement_mesh<tf::none_t>(graph);
}

template <typename Graph>
auto arrangement_curves_of(const Graph &graph) -> arrangement_curves_t<Graph> {
  return tf::make_intersection_curves<tf::none_t>(graph);
}

template <typename Forms>
auto mesh_arrangements_of(Forms forms, tf::arrangement_config config)
    -> mesh_arrangements_result_t<Forms> {
  return tf::make_mesh_arrangements<tf::none_t, tf::none_t>(std::move(forms),
                                                            config);
}

template <typename Forms>
auto mesh_arrangements_with_curves_of(Forms forms,
                                      tf::arrangement_config config)
    -> mesh_arrangements_curves_result_t<Forms> {
  return tf::make_mesh_arrangements<tf::none_t, tf::none_t>(
      std::move(forms), config, tf::return_curves);
}

template <typename Form>
auto polygon_arrangements_of(const Form &form, tf::arrangement_config config)
    -> polygon_arrangements_result_t<Form> {
  return tf::make_polygon_arrangements<tf::none_t, tf::none_t>(form, config);
}

template <typename Form>
auto polygon_arrangements_with_curves_of(const Form &form,
                                         tf::arrangement_config config)
    -> polygon_arrangements_curves_result_t<Form> {
  return tf::make_polygon_arrangements<tf::none_t, tf::none_t>(
      form, config, tf::return_curves);
}

} // namespace tf::test
