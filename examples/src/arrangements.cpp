// Booleans and domains from one csg_graph build.
//
// One scene -- a sphere straddling a knife plane declared a sheet, plus
// two floaters that touch nothing -- queried three ways: the sides as
// boolean meshes, the volumes individually via expression-selected
// domains, and the same selection by hand from the inclusion matrix.
//
// Mirrors python/examples/arrangements.py on the Python side.

#include <trueform/trueform.hpp>

#include <array>
#include <iostream>
#include <string>
#include <vector>

int main() {
  using Index = int;
  using Real = float;

  // -------------------------------------------------------------------
  // 1. Geometry: a sphere straddling z=0, one floater above, one below,
  //    and a 4x4 knife plane on z=0 with stored normal +Z.
  // -------------------------------------------------------------------
  auto straddle = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto above_s = tf::make_sphere_mesh<Index>(Real(0.5), 32, 32);
  auto below_s = tf::make_sphere_mesh<Index>(Real(0.5), 32, 32);
  auto plane = tf::make_plane_mesh<Index>(Real(4), Real(4));

  auto fid = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(0), Real(0), Real(0)}));
  auto fa = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(0), Real(0), Real(2)}));
  auto fb = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(0), Real(0), Real(-2)}));

  auto p0 = straddle.polygons() | tf::tag(fid);  // op 0: straddles
  auto p1 = above_s.polygons() | tf::tag(fa);    // op 1: floats above
  auto p2 = below_s.polygons() | tf::tag(fb);    // op 2: floats below
  auto p_knife = plane.polygons() | tf::tag(fid);  // op 3: the knife

  std::vector<decltype(p0)> forms{p0, p1, p2, p_knife};

  // -------------------------------------------------------------------
  // 2. One build. Declaring the plane a sheet makes operand 3 an
  //    oriented separator: its bit means "behind the sheet's normal"
  //    (-Z here), so the knife cuts volumes without enclosing one.
  // -------------------------------------------------------------------
  std::array<int, 1> sheets{3};
  auto graph =
      tf::make_csg_graph(tf::make_range(forms), tf::make_range(sheets));
  auto solids = tf::csg::merge(tf::csg::merge(0, 1), 2);

  // -------------------------------------------------------------------
  // 3. The boolean path: one mesh per side. Each comes back closed --
  //    the straddler's half capped by the knife, the floater whole --
  //    as disjoint pieces of a single mesh.
  // -------------------------------------------------------------------
  auto above_mesh = tf::make_csg_mesh(graph, tf::csg::difference(solids, 3));
  auto below_mesh =
      tf::make_csg_mesh(graph, tf::csg::intersection(solids, 3));
  std::cout << "=== Boolean meshes ===" << std::endl;
  std::cout << "  solids - knife: vol="
            << tf::signed_volume(above_mesh.polygons())
            << " closed=" << tf::is_closed(above_mesh.polygons())
            << std::endl;
  std::cout << "  solids & knife: vol="
            << tf::signed_volume(below_mesh.polygons())
            << " closed=" << tf::is_closed(below_mesh.polygons())
            << std::endl;

  // -------------------------------------------------------------------
  // 4. Domains by expression: the same volumes, individually.
  // -------------------------------------------------------------------
  auto [above_cells, above_ids] =
      tf::make_csg_domains(graph, tf::csg::difference(solids, 3));
  auto [below_cells, below_ids] =
      tf::make_csg_domains(graph, tf::csg::intersection(solids, 3));
  std::cout << "\n=== Domains by expression ===" << std::endl;
  std::cout << "  above: " << above_cells.size() << " cells" << std::endl;
  std::cout << "  below: " << below_cells.size() << " cells" << std::endl;

  // -------------------------------------------------------------------
  // 5. Domains by hand: extract everything once, select by mask. The
  //    knife's column is 3; behind its +Z normal = below.
  // -------------------------------------------------------------------
  auto [cells, ids, imap] =
      tf::make_csg_domains(graph, tf::return_index_map);
  tf::buffer<char> below;
  below.allocate(cells.size());
  std::size_t n_below = 0;
  for (std::size_t k = 0; k < cells.size(); ++k) {
    below[k] = imap.inclusion[k][3] ? char(1) : char(0);
    n_below += static_cast<std::size_t>(below[k]);
  }
  std::cout << "\n=== Domains by hand ===" << std::endl;
  std::cout << "  " << cells.size() << " cells; above "
            << cells.size() - n_below << ", below " << n_below << std::endl;

  // -------------------------------------------------------------------
  // 6. Write and verify. Each cell is closed and manifold by
  //    construction.
  // -------------------------------------------------------------------
  auto write_side = [&, &cs = cells](bool want_below, const char *prefix) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < cs.size(); ++i) {
      if (bool(below[i]) != want_below)
        continue;
      const auto &cell = cs[i];
      auto fname = std::string(prefix) + "_" + std::to_string(k++) + ".stl";
      tf::write_stl(cell.polygons(), fname);
      std::cout << "  wrote " << fname << " (faces=" << cell.faces().size()
                << ", closed=" << tf::is_closed(cell.polygons())
                << ", manifold=" << tf::is_manifold(cell.polygons()) << ")"
                << std::endl;
    }
  };
  write_side(false, "above");
  write_side(true, "below");

  return 0;
}
