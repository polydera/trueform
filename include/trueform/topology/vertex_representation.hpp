/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/faces.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_membership.hpp"

namespace tf {
namespace topology {

template <typename Index> struct vertex_representation_inner_dref {

  template <typename T> auto operator()(T &&v_id) const {
    return (*_fe_ptr)[v_id].front() == face_id;
  }
  Index face_id;
  const tf::face_membership<Index> *_fe_ptr;
};

template <typename Index> struct vertex_representation_dref {
  const tf::face_membership<Index> *_fe_ptr;
  //

  //
  template <typename T> auto operator()(T &&pair) const {
    auto &&[face_id, face] = pair;
    return tf::make_mapped_range(
        face, vertex_representation_inner_dref<Index>{Index(face_id), _fe_ptr});
  }
};
} // namespace topology

template <typename Index, typename Range>
auto make_vertex_representation(Index face_id, const Range &face,
                                const tf::face_membership<Index> &fe) {
  return tf::make_mapped_range(
      face,
      topology::vertex_representation_inner_dref<Index>{Index(face_id), &fe});
}

template <typename Policy, typename Index>
auto make_vertex_representation(const tf::faces<Policy> &faces,
                                const tf::face_membership<Index> &fe) {
  return tf::make_mapped_range(
      tf::enumerate(faces), topology::vertex_representation_dref<Index>{&fe});
}
} // namespace tf
