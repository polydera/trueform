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
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto finalize_delaunay_index_map(Owner &owner, std::size_t n_input) -> void {
  using Index = typename Owner::index_type;
  owner._delaunay_representative_to_unique.allocate_and_initialize(n_input,
                                                                   Index(-1));
  owner._index_map.kept_ids().allocate(owner._sites.size());
  for (std::size_t vertex = 0; vertex < owner._sites.size(); ++vertex) {
    const Index input = owner._sites[vertex].output;
    owner._delaunay_representative_to_unique[std::size_t(input)] =
        static_cast<Index>(vertex);
    owner._index_map.kept_ids()[vertex] = input;
  }
  for (std::size_t input = 0; input < n_input; ++input) {
    const Index representative = owner._index_map.f()[input];
    owner._index_map.f()[input] =
        owner._delaunay_representative_to_unique[std::size_t(representative)];
  }
}

} // namespace tf::topology::cdt
