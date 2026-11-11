/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/config/verify.hpp"
#include "../../core/range.hpp"
#include "../../topology/cdt/dt3/check_tetrahedralization_validity.hpp"
#include "../../topology/cdt/dt3/clear_tet_written_rows.hpp"
#include "../../topology/cdt/dt3/recovery/restore_tet_delaunay.hpp"
#include "../../topology/cdt/dt3/recovery/tet_kept_rows.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_recovery.hpp"
#include "../regions/carry_tet_regions.hpp"
#include "../regions/tet_regions.hpp"
#include "../regions/wind_tet_regions.hpp"
#include <cstddef>

namespace tf::tetmesh {

/// Restore the region-labeled complex `owner` to Delaunay wherever flips reach
/// among the cells `regions` keeps: the rows the recovery recorded that the
/// region labeling winds positively seed the restoration (@ref
/// restore_tet_delaunay), the segments of `recovery` barred. Its flips cross no
/// protected facet, so every winding stands and the rows they write take theirs
/// from the cells they meet (@ref carry_tet_regions); a region whose every row
/// was replaced is labeled again over the covering triangles `faces`
/// (@ref wind_tet_regions). The complex is published canonically, so the
/// cells a reader sees follow their rank and not the restoration's writing
/// order. The rows are recorded no further. Returns the recorded edges no
/// flip removed.
template <typename Owner, typename Faces, typename Coord>
auto restore_tet_kept_cells(Owner &owner, const Faces &faces,
                            tet_regions<typename Owner::index_type> &regions,
                            tf::topology::cdt::dt3::tet_segment_recovery<
                                typename Owner::index_type, Coord> &recovery)
    -> std::size_t {
  using Index = typename Owner::index_type;
  const std::size_t n_finite = owner._n_finite;
  const std::size_t n_rows = owner._corners.size();
  const int *winding = regions.winding.begin();
  const std::size_t left = tf::topology::cdt::dt3::restore_tet_delaunay(
      owner, recovery,
      tf::topology::cdt::dt3::tet_kept_rows<Index>{
          tf::make_range(winding, winding + n_finite), true});
  owner._record_written = false;
  tf::topology::cdt::dt3::clear_tet_written_rows(owner);
  if (carry_tet_regions(owner, regions, n_finite, n_rows))
    wind_tet_regions(owner, faces, regions);
  if constexpr (tf::core::verification) {
    tf::core::verify(
        tf::topology::cdt::dt3::check_tetrahedralization_validity(owner),
        "the restoration of the kept cells publishes a valid complex");
    tf::core::verify(owner._canonical,
                     "the restoration of the kept cells publishes the rank");
    tet_regions<Index> stated;
    wind_tet_regions(owner, faces, stated);
    bool carried =
        stated.consistent && stated.winding.size() == regions.winding.size();
    for (std::size_t cell = 0; carried && cell < stated.winding.size(); ++cell)
      carried = stated.winding[cell] == regions.winding[cell];
    tf::core::verify(carried, "the carried winding is the labeling's");
  }
  return left;
}

} // namespace tf::tetmesh
