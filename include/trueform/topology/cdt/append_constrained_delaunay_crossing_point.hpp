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
#include "./blend_constrained_delaunay_edge.hpp"
#include "./locate_constrained_delaunay.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Index, typename Location>
struct constrained_delaunay_crossing_point {
  Index vertex;
  Index coincident;
  Location location;
};

/// A created crossing extends the exact point and output-identity carriers in
/// lockstep. `index_map().f().size()` is the synthetic-vertex sentinel in the
/// inverse map.
template <typename Owner>
auto append_constrained_delaunay_crossing_point(
    Owner &owner, typename Owner::index_type first,
    typename Owner::index_type second, typename Owner::param_type parameter)
    -> constrained_delaunay_crossing_point<typename Owner::index_type,
                                           typename Owner::locate_result> {
  using Index = typename Owner::index_type;
  constrained_delaunay_crossing_point<Index, typename Owner::locate_result>
      result{Owner::none, Owner::none, {}};
  const auto point =
      blend_constrained_delaunay_edge(owner, first, second, parameter);
  result.location = locate_constrained_delaunay(owner, point);
  if (result.location.he != Owner::none) {
    Index edge = result.location.he;
    for (int corner = 0; corner < 3 && edge != Owner::none; ++corner) {
      const auto candidate = owner._points[std::size_t(owner.origin(edge))];
      if (candidate[0] == point[0] && candidate[1] == point[1]) {
        result.coincident = owner.origin(edge);
        return result;
      }
      edge = owner.previous_edge(Owner::opposite(edge));
    }
  }
  if (owner._points.size() >= Owner::k_max_topology_sites)
    return result;
  result.vertex = static_cast<Index>(owner._points.size());
  owner._points.push_back(point);
  owner._index_map.kept_ids().push_back(
      static_cast<Index>(owner._index_map.f().size()));
  owner._v_first_edge.reallocate(owner._points.size());
  owner._v_first_edge[std::size_t(result.vertex)] = Owner::none;
  return result;
}

} // namespace tf::topology::cdt
