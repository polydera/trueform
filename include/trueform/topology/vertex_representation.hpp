/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/faces.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/mapped_range.hpp"
#include "./face_membership_like.hpp"

namespace tf {
namespace topology {

template <typename Policy> struct vertex_representation_inner_dref {

  template <typename T> auto operator()(T &&v_id) const {
    return std::size_t(_fe[v_id].front()) == face_id;
  }
  std::size_t face_id;
  tf::face_membership_like<Policy> _fe;
};

template <typename Policy> struct vertex_representation_dref {
  tf::face_membership_like<Policy> _fe;
  //
  template <typename T> auto operator()(T &&pair) const {
    auto &&[face_id, face] = pair;
    return tf::make_mapped_range(face, vertex_representation_inner_dref<Policy>{
                                           std::size_t(face_id), _fe});
  }
};
} // namespace topology

template <typename Range, typename Policy>
auto make_vertex_representation(std::size_t face_id, const Range &face,
                                const tf::face_membership_like<Policy> &fe) {
  auto r = tf::make_range(fe);
  return tf::make_mapped_range(
      face, topology::vertex_representation_inner_dref<decltype(r)>{
                face_id, tf::make_face_membership_like(std::move(r))});
}

template <typename Policy0, typename Policy1>
auto make_vertex_representation(const tf::faces<Policy0> &faces,
                                const tf::face_membership_like<Policy1> &fe) {
  auto r = tf::make_range(fe);
  return tf::make_mapped_range(
      tf::enumerate(faces), topology::vertex_representation_dref<decltype(r)>{
                                tf::make_face_membership_like(std::move(r))});
}
} // namespace tf
