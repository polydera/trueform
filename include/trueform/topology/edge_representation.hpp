/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/views/enumerate.hpp"
#include "../core/views/mapped_range.hpp"
#include "./manifold_edge_link_like.hpp"

namespace tf {
namespace topology {

template <typename Index> struct edge_representation_inner_dref {

  template <typename T> auto operator()(T &&fp) const {
    return fp.is_representative(face_id);
  }
  Index face_id;
};

template <typename Index> struct edge_representation_dref {
  template <typename T> auto operator()(T &&pair) const {
    auto &&[id, link] = pair;
    return tf::make_mapped_range(
        link, edge_representation_inner_dref<Index>{Index(id)});
  }
};
} // namespace topology

template <typename Index, typename Policy>
auto make_edge_representation(Index face_id,
                              const tf::manifold_edge_link_like<Policy> &mel) {
  return tf::make_mapped_range(
      mel[face_id],
      topology::edge_representation_inner_dref<Index>{Index(face_id)});
}

template <typename Policy>
auto make_edge_representation(const tf::manifold_edge_link_like<Policy> &mel) {
  using Index = std::decay_t<decltype(mel[0][0].face_peer)>;
  return tf::make_mapped_range(tf::enumerate(mel),
                               topology::edge_representation_dref<Index>{});
}
} // namespace tf
