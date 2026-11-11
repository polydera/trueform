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
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/tetmesh/regions/carry_tet_regions.hpp>
#include <trueform/tetmesh/regions/tet_regions.hpp>
#include <trueform/tetmesh/regions/wind_tet_regions.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/protect_tet_facets.hpp>
#include <trueform/topology/cdt/dt3/publish_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/tet_state.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using cry_owner_t =
    tf::topology::cdt::dt3::tetrahedralization_owner<int, std::int64_t,
                                                     tf::exact::int32>;

/// A tetrahedron's four sites on the lattice, its own four faces wound
/// outward. The hull facets of a point set in convex position are facets of
/// its Delaunay complex, so every face is found and protected, which leaves
/// the region labeling two regions: the one finite cell the faces enclose, and
/// the hull cells outside them.
auto cry_scene() -> std::vector<std::int64_t> {
  return {0, 0, 0, 4096, 0, 0, 0, 4096, 0, 0, 0, 4096};
}

auto cry_faces() -> std::vector<std::array<int, 3>> {
  return {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
}

/// The scene built, published and protected face by face, as the mesher's
/// own phases leave a complex before the region labeling reads it.
auto cry_build(cry_owner_t &owner) -> bool {
  const auto flat = cry_scene();
  if (!tf::topology::cdt::dt3::build_tetrahedralization(
          owner, tf::make_points<3>(tf::make_range(flat))))
    return false;
  tf::topology::cdt::dt3::publish_tetrahedralization<true>(owner);
  const auto slot = owner._index_map.f();
  const auto faces = cry_faces();
  for (std::size_t f = 0; f < faces.size(); ++f) {
    const std::array<int, 3> facet{slot[std::size_t(faces[f][0])],
                                   slot[std::size_t(faces[f][1])],
                                   slot[std::size_t(faces[f][2])]};
    const std::vector<std::array<int, 3>> one{facet};
    tf::topology::cdt::dt3::protect_tet_facets(owner, one, int(f));
  }
  return true;
}

/// The covering triangles in site slots, which the region labeling reads the
/// crossings through.
auto cry_covering(const cry_owner_t &owner) -> std::vector<std::array<int, 3>> {
  const auto slot = owner._index_map.f();
  auto faces = cry_faces();
  for (auto &face : faces)
    for (std::size_t k = 0; k < 3; ++k)
      face[k] = slot[std::size_t(face[k])];
  return faces;
}

} // namespace

TEST_CASE("carry_tet_regions: a region that kept no row is labeled again",
          "[tetmesh][regions]") {
  // The branch fires where a kept region holds no row below the restoration's
  // first, so the closure over unprotected facets reaches none of its cells.
  // No surface of the corpus and no five-site sweep reaches that through
  // flips, so the state is stated through the carry's own parameters: the
  // rows the regions cover and the rows it was stated over are the carry's
  // two counts, and passing both as zero says every row of every region is
  // one the flips wrote. The complex is the one the phases publish, built
  // and protected through the owner-level calls alone.
  namespace dt3 = tf::topology::cdt::dt3;
  cry_owner_t forced, plain;
  REQUIRE(cry_build(forced));
  REQUIRE(cry_build(plain));
  REQUIRE(dt3::check_tetrahedralization_validity(forced));
  const auto covering = cry_covering(forced);

  // the oracle: the scene is labeled a kept region and an outside one, so
  // the branch has something to lose
  tf::tetmesh::tet_regions<int> stated;
  tf::tetmesh::wind_tet_regions(forced, covering, stated);
  REQUIRE(stated.consistent);
  REQUIRE(stated.winding.size() == std::size_t(forced._n_finite));
  std::size_t kept = 0;
  for (const int winding : stated.winding)
    kept += winding > 0;
  REQUIRE(kept > 0);

  // the unforced path on the twin: the carry carries what the region labeling
  // stated
  tf::tetmesh::tet_regions<int> carried;
  tf::tetmesh::wind_tet_regions(plain, covering, carried);
  REQUIRE(carried.consistent);
  const std::size_t unreached_plain = tf::tetmesh::carry_tet_regions(
      plain, carried, std::size_t(plain._n_finite),
      std::size_t(plain._corners.size()));
  CHECK(unreached_plain == 0);

  // (a) the branch is taken: no row carries a winding, so the carry reaches
  // none of the live rows and hands them back
  tf::tetmesh::tet_regions<int> lost;
  tf::tetmesh::wind_tet_regions(forced, covering, lost);
  REQUIRE(lost.consistent);
  const std::size_t unreached = tf::tetmesh::carry_tet_regions(
      forced, lost, std::size_t(0), std::size_t(0));
  CHECK(unreached > 0);

  // (b) the region labeling states the winding again, cell for cell
  tf::tetmesh::wind_tet_regions(forced, covering, lost);
  REQUIRE(lost.consistent);
  REQUIRE(lost.winding.size() == carried.winding.size());
  for (std::size_t cell = 0; cell < lost.winding.size(); ++cell)
    CHECK(lost.winding[cell] == carried.winding[cell]);

  // (c) the product is the one the unforced path publishes
  REQUIRE(dt3::check_tetrahedralization_validity(forced));
  CHECK(forced._canonical);
  CHECK(plain._canonical);
  REQUIRE(forced._corners.size() == plain._corners.size());
  REQUIRE(std::size_t(forced._n_finite) == std::size_t(plain._n_finite));
  for (std::size_t row = 0; row < std::size_t(forced._corners.size()); ++row) {
    CHECK((forced._states[row] == dt3::tet_state::dead) ==
          (plain._states[row] == dt3::tet_state::dead));
    for (std::size_t k = 0; k < 4; ++k)
      CHECK(forced._corners[row][k] == plain._corners[row][k]);
  }
}
