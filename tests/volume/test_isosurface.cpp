#include "../common/arrangement_builders.hpp"
#include "../common/arrangement_readers.hpp"
#include "../common/csg_readers.hpp"
#include "../common/tagged_operand.hpp"
#include "../common/volume_generators.hpp"

#include <catch2/catch_test_macros.hpp>

#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg/boolean_op.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/volume/impl/dual_contour_tables.hpp>
#include <trueform/volume/impl/flying_dc.hpp>
#include <trueform/volume/make_isosurface.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace field = tf::test::volume_field;

struct surface_report {
  std::size_t boundary_edges = 0;
  std::size_t over_incident = 0;
  std::size_t pinched_vertices = 0;
  std::size_t degenerate_faces = 0;
  std::size_t unpaired_orientation = 0;
};

// Edge degree is not the whole of manifoldness: a vertex whose incident corners
// form two cycles is a pinch, and a surface whose directed edges do not pair is
// not consistently oriented.
struct corner_link_pair {
  int v, a, b;
};

auto pinched_count(const std::vector<corner_link_pair> &links) -> std::size_t {
  std::map<int, std::vector<std::pair<int, int>>> around;
  for (const auto &l : links)
    around[l.v].push_back({l.a, l.b});
  std::size_t pinched = 0;
  for (const auto &[v, fan] : around) {
    static_cast<void>(v);
    std::map<int, int> parent;
    const std::function<int(int)> find = [&](int x) {
      return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    for (const auto &[a, b] : fan) {
      if (!parent.count(a))
        parent[a] = a;
      if (!parent.count(b))
        parent[b] = b;
    }
    for (const auto &[a, b] : fan) {
      const int ra = find(a), rb = find(b);
      if (ra != rb)
        parent[rb] = ra;
    }
    std::size_t components = 0;
    for (const auto &node : parent)
      components += find(node.first) == node.first;
    pinched += components > 1;
  }
  return pinched;
}

template <typename Polygons>
auto inspect(const Polygons &polys) -> surface_report {
  surface_report r;
  std::map<std::pair<int, int>, int> edges;
  std::map<std::pair<int, int>, int> directed;
  std::vector<corner_link_pair> links;
  for (const auto &poly : polys) {
    const int m = int(poly.size());
    for (int i = 0; i < m; ++i) {
      const int a = int(poly.indices()[i]);
      const int b = int(poly.indices()[(i + 1) % m]);
      const int prev = int(poly.indices()[(i + m - 1) % m]);
      if (a == b)
        ++r.degenerate_faces;
      ++edges[{std::min(a, b), std::max(a, b)}];
      ++directed[{a, b}];
      links.push_back({a, prev, b});
    }
  }
  for (const auto &[e, d] : edges) {
    r.boundary_edges += d == 1;
    r.over_incident += d > 2;
  }
  for (const auto &[d, n] : directed)
    if (!directed.count({d.second, d.first}))
      r.unpaired_orientation += std::size_t(n);
  r.pinched_vertices = pinched_count(links);
  return r;
}

} // namespace

TEST_CASE("the dual contour case tables describe disks",
          "[volume][isosurface]") {
  using namespace tf::volume_detail;
  REQUIRE_FALSE(patches().malformed);
  int instances = 0;
  for (std::size_t cs = 0; cs < 256; ++cs)
    for (int comp = 0; comp < components().count[cs]; ++comp) {
      ++instances;
      const int len = patches().link_len[cs][std::size_t(comp)];
      REQUIRE(len > 0);
      // the boundary link visits every crossing of the component once
      int seen[12] = {};
      for (int i = 0; i < len; ++i)
        ++seen[patches().link[cs][std::size_t(comp)][std::size_t(i)]];
      for (int e = 0; e < 12; ++e)
        REQUIRE(seen[e] ==
                (components().comp[cs][std::size_t(e)] == comp ? 1 : 0));
    }
  REQUIRE(instances == 358);
}

TEST_CASE("neighbouring cells agree on the arcs of the face they share",
          "[volume][isosurface]") {
  using namespace tf::volume_detail;
  int collisions = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int low = axis * 2, high = axis * 2 + 1;
    for (int lower = 0; lower < 256; ++lower)
      for (int free_bits = 0; free_bits < 16; ++free_bits) {
        int upper = 0;
        for (int i = 0; i < 4; ++i)
          if ((lower >> k_face_corners[std::size_t(high)][std::size_t(i)]) & 1)
            upper |= 1 << k_face_corners[std::size_t(low)][std::size_t(i)];
        int slot = 0;
        for (int c = 0; c < 8; ++c) {
          bool on_face = false;
          for (int i = 0; i < 4; ++i)
            on_face = on_face ||
                      k_face_corners[std::size_t(low)][std::size_t(i)] == c;
          if (on_face)
            continue;
          if ((free_bits >> slot) & 1)
            upper |= 1 << c;
          ++slot;
        }
        const auto lc = static_cast<unsigned char>(lower);
        const auto uc = static_cast<unsigned char>(upper);
        const int n = patches().arc_count[lc][std::size_t(high)];
        REQUIRE(n == patches().arc_count[uc][std::size_t(low)]);
        for (int i = 0; i < n; ++i)
          REQUIRE(patches().arc_pair[lc][std::size_t(high)][std::size_t(i)] ==
                  patches().arc_pair[uc][std::size_t(low)][std::size_t(i)]);
        if (n == 2 &&
            patches().arc_comp[lc][std::size_t(high)][0] ==
                patches().arc_comp[lc][std::size_t(high)][1] &&
            patches().arc_comp[uc][std::size_t(low)][0] ==
                patches().arc_comp[uc][std::size_t(low)][1])
          ++collisions;
      }
  }
  // the configuration the completion exists for must be in the table
  REQUIRE(collisions > 0);
}

TEST_CASE("a cut boundary link splits into separate vertices",
          "[volume][isosurface]") {
  using namespace tf::volume_detail;
  // at x = 0 the quads of edges 4, 6, 8 and 10 do not exist
  const unsigned retained = retained_edges(0, 4, 4, 16, 16, 16);
  REQUIRE(retained ==
          (k_all_edges & ~((1u << 4) | (1u << 6) | (1u << 8) | (1u << 10))));
  const auto cs = static_cast<unsigned char>(62);
  REQUIRE(cell_vertex_count(cs, k_all_edges) == components().count[cs]);
  REQUIRE(cell_vertex_count(cs, retained) == 2);
  REQUIRE(cell_vertex_index(cs, retained, 7) ==
          cell_vertex_index(cs, retained, 11));
  REQUIRE(cell_vertex_index(cs, retained, 0) !=
          cell_vertex_index(cs, retained, 7));
}

// The completion's claim is about the polygon surface the triangles tessellate:
// quads, and 5-8-gons where a face's two arcs were given distinct identities.
// Production writes triangles directly, so the builder states that surface only
// when a consumer asks — and this is the consumer.
TEST_CASE("the completed polygon surface is manifold", "[volume][isosurface]") {
  const auto audit_polygons = [](const auto &vol, bool refine) {
    tf::volume_detail::dual_contouring<int, float> dc;
    dc.record_polygons(true);
    tf::isosurface_config config;
    config.method = tf::isosurface_method::dual_contouring;
    config.refine = refine;
    const auto mesh = dc.build(vol.volume(), 0.f, config);

    // the polygons in the library's own polygon type, so the real predicates
    // answer for them
    tf::polygons_buffer<int, float, 3, tf::dynamic_size> polys;
    polys.points_buffer() = mesh.points_buffer();
    auto &faces = polys.faces_buffer();
    faces.offsets_buffer().allocate(dc.polygon_offsets().size());
    std::memcpy(faces.offsets_buffer().data(), dc.polygon_offsets().data(),
                dc.polygon_offsets().size() * sizeof(int));
    faces.data_buffer().allocate(dc.polygon_indices().size());
    std::memcpy(faces.data_buffer().data(), dc.polygon_indices().data(),
                dc.polygon_indices().size() * sizeof(int));
    return polys;
  };

  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;
  for (const bool refine : {true, false}) {
    config.refine = refine;
    // resolutions where two arcs of one face are known to meet the same pair
    // of cells: without the completion these carry over-incident edges
    for (const int n : {32, 48, 64}) {
      for (const float angle : {30.f, 45.f, 90.f, 120.f}) {
        INFO("wedge " << angle << " n = " << n << " refine = " << refine);
        const auto vol = tf::test::sampled_volume(field::wedge(angle), n);
        const auto polys = audit_polygons(vol, refine);
        REQUIRE(polys.polygons().size() > 0);
        REQUIRE(tf::is_manifold(polys.polygons()));
        REQUIRE(tf::is_closed(polys.polygons()));
        const auto r = inspect(polys.polygons());
        REQUIRE(r.over_incident == 0);
        REQUIRE(r.pinched_vertices == 0);
        REQUIRE(r.degenerate_faces == 0);
        REQUIRE(r.unpaired_orientation == 0);
        // the triangles are exactly this surface tessellated: a quad gives
        // two, a subdivided polygon gives one per side around its fan centre
        std::size_t implied = 0;
        for (const auto &poly : polys.polygons()) {
          REQUIRE(poly.size() >= 4);
          REQUIRE(poly.size() <= 8);
          implied += poly.size() == 4 ? 2 : poly.size();
        }
        REQUIRE(
            implied ==
            std::size_t(tf::make_isosurface(vol.volume(), 0.f, config).polygons().size()));
      }
      {
        INFO("box n = " << n << " refine = " << refine);
        const auto polys =
            audit_polygons(tf::test::sampled_volume(field::box{}, n), refine);
        REQUIRE(tf::is_manifold(polys.polygons()));
        REQUIRE(tf::is_closed(polys.polygons()));
        const auto r = inspect(polys.polygons());
        REQUIRE(r.over_incident == 0);
        REQUIRE(r.pinched_vertices == 0);
      }
      {
        // the adversarial field: every case, and the collisions that go with it
        INFO("noise n = " << n << " refine = " << refine);
        const auto polys =
            audit_polygons(tf::test::sampled_volume(field::noise{}, n), refine);
        REQUIRE(tf::is_manifold(polys.polygons()));
        const auto r = inspect(polys.polygons());
        REQUIRE(r.over_incident == 0);
        REQUIRE(r.pinched_vertices == 0);
        REQUIRE(r.degenerate_faces == 0);
      }
    }
  }
}

// The recording requests select other instantiations of the emit and the refit
// passes, and production takes the ones that record nothing. A claim a
// recording build proves is a claim about the shipped path only if the two emit
// the same surface, byte for byte.
TEST_CASE("a recording build is the shipped build", "[volume][isosurface]") {
  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;
  for (const bool refine : {true, false}) {
    config.refine = refine;
    for (const int n : {24, 32, 48}) {
      INFO("n = " << n << " refine = " << refine);
      const auto vol = tf::test::sampled_volume(field::wedge(45.f), n);
      tf::volume_detail::dual_contouring<int, float> dc;
      dc.record_polygons(true);
      dc.record_refit_provenance(true);
      const auto recorded = dc.build(vol.volume(), 0.f, config);
      const auto shipped = tf::make_isosurface(vol.volume(), 0.f, config);
      const auto &rp = recorded.points_buffer().data_buffer();
      const auto &sp = shipped.points_buffer().data_buffer();
      const auto &rf = recorded.faces_buffer().data_buffer();
      const auto &sf = shipped.faces_buffer().data_buffer();
      REQUIRE(rp.size() == sp.size());
      REQUIRE(rf.size() == sf.size());
      REQUIRE(rf.size() > 0u);
      REQUIRE(std::memcmp(rp.data(), sp.data(), rp.size() * sizeof(float)) == 0);
      REQUIRE(std::memcmp(rf.data(), sf.data(), rf.size() * sizeof(int)) == 0);
    }
  }
}

TEST_CASE("a quad side names the cell that owns the face it crosses",
          "[volume][isosurface]") {
  using namespace tf::volume_detail;
  // cell offsets of the four corners of each quad type, in emitted order
  const int corner[3][4][3] = {
      {{0, -1, -1}, {0, 0, -1}, {0, 0, 0}, {0, -1, 0}},
      {{-1, 0, -1}, {-1, 0, 0}, {0, 0, 0}, {0, 0, -1}},
      {{-1, -1, 0}, {0, -1, 0}, {0, 0, 0}, {-1, 0, 0}}};
  for (int axis = 0; axis < 3; ++axis)
    for (int s = 0; s < 4; ++s) {
      const auto &side = k_quad_sides[std::size_t(axis)][std::size_t(s)];
      const int *a = corner[axis][s];
      const int *b = corner[axis][(s + 1) % 4];
      int differing = -1, n_diff = 0;
      for (int d = 0; d < 3; ++d)
        if (a[d] != b[d]) {
          differing = d;
          ++n_diff;
        }
      REQUIRE(n_diff == 1);
      REQUIRE(differing == int(side.face_axis));
      // the owner is the lower cell of the pair, which is what makes one cell
      // decide the face for all four quads around it
      const int *owner = corner[axis][side.owner_corner];
      REQUIRE(owner[differing] == std::min(a[differing], b[differing]));
      // and the owned edge runs along the quad's own axis
      const int edge =
          k_face_edges[std::size_t(side.face_axis) * 2 + 1][side.ordinal];
      const auto &e0 =
          k_corner_offset[std::size_t(k_cell_edge[std::size_t(edge)][0])];
      const auto &e1 =
          k_corner_offset[std::size_t(k_cell_edge[std::size_t(edge)][1])];
      int moved = -1;
      for (int d = 0; d < 3; ++d)
        if (e0[std::size_t(d)] != e1[std::size_t(d)])
          moved = d;
      REQUIRE(moved == axis);
    }
}

TEST_CASE("dual contouring of a closed field is closed and manifold",
          "[volume][isosurface]") {
  const auto config =
      tf::isosurface_config(tf::isosurface_method::dual_contouring);
  SECTION("box") {
    const auto vol = tf::test::sampled_volume(field::box{}, 32);
    const auto mesh = tf::make_isosurface(vol.volume(), 0.f, config);
    const auto polys = mesh.polygons();
    REQUIRE(polys.size() > 0);
    REQUIRE(tf::is_manifold(polys));
    REQUIRE(tf::is_closed(polys));
    const auto r = inspect(polys);
    REQUIRE(r.over_incident == 0);
    REQUIRE(r.pinched_vertices == 0);
    REQUIRE(r.degenerate_faces == 0);
    REQUIRE(r.unpaired_orientation == 0);
    // outward winding, like flying edges and marching cubes
    REQUIRE(double(tf::signed_volume(polys)) > 0);
  }
  SECTION("wedges, where two contour arcs can share a face") {
    for (const float angle : {30.f, 45.f, 60.f, 90.f, 120.f}) {
      const auto vol = tf::test::sampled_volume(field::wedge(angle), 48);
      const auto mesh = tf::make_isosurface(vol.volume(), 0.f, config);
      const auto polys = mesh.polygons();
      REQUIRE(polys.size() > 0);
      REQUIRE(tf::is_manifold(polys));
      REQUIRE(tf::is_closed(polys));
      const auto r = inspect(polys);
      REQUIRE(r.over_incident == 0);
      REQUIRE(r.pinched_vertices == 0);
      REQUIRE(r.unpaired_orientation == 0);
    }
  }
  SECTION("a pseudo-random sign field, which hits every case") {
    const auto vol = tf::test::sampled_volume(field::noise{}, 48);
    const auto mesh = tf::make_isosurface(vol.volume(), 0.f, config);
    const auto polys = mesh.polygons();
    REQUIRE(polys.size() > 0);
    REQUIRE(tf::is_manifold(polys));
    const auto r = inspect(polys);
    REQUIRE(r.over_incident == 0);
    REQUIRE(r.pinched_vertices == 0);
    REQUIRE(r.degenerate_faces == 0);
    REQUIRE(r.unpaired_orientation == r.boundary_edges);
  }
}

TEST_CASE("a surface that leaves the domain is manifold with boundary",
          "[volume][isosurface]") {
  const auto vol = tf::test::sampled_volume(field::slab{}, 48);
  const auto mesh =
      tf::make_isosurface(vol.volume(), 0.f, tf::isosurface_method::dual_contouring);
  const auto polys = mesh.polygons();
  REQUIRE(polys.size() > 0);
  REQUIRE(tf::is_manifold(polys));
  REQUIRE_FALSE(tf::is_closed(polys));
  const auto r = inspect(polys);
  REQUIRE(r.over_incident == 0);
  REQUIRE(r.pinched_vertices == 0);
  REQUIRE(r.boundary_edges > 0);
}

TEST_CASE("dual contouring places its vertices on the surface",
          "[volume][isosurface]") {
  const field::box shape;
  const auto vol = tf::test::sampled_volume(shape, 48);
  const auto fe = tf::make_isosurface(vol.volume());
  const auto dc =
      tf::make_isosurface(vol.volume(), 0.f, tf::isosurface_method::dual_contouring);
  const double truth = 8.0 * 0.62 * 0.45 * 0.33;
  const double fe_volume = double(tf::signed_volume(fe.polygons()));
  const double dc_volume = double(tf::signed_volume(dc.polygons()));
  // both wound outward, and the fitted vertices recover the corners the
  // grid-edge vertices round off
  REQUIRE(fe_volume > 0);
  REQUIRE(dc_volume > 0);
  REQUIRE(std::abs(dc_volume - truth) < std::abs(fe_volume - truth));
}

TEST_CASE("the isosurface method is a runtime choice", "[volume][isosurface]") {
  const auto vol = tf::test::sampled_volume(field::sphere{}, 24);
  const auto plain = tf::make_isosurface(vol.volume());
  const auto configured =
      tf::make_isosurface(vol.volume(), 0.f, tf::isosurface_method::flying_edges);
  REQUIRE(plain.polygons().size() == configured.polygons().size());
  const auto dual =
      tf::make_isosurface(vol.volume(), 0.f, tf::isosurface_method::dual_contouring);
  REQUIRE(dual.polygons().size() > 0);
  REQUIRE(dual.points().size() != plain.points().size());
}

// A vertex is a coordinate before it is anything else. A cell whose component
// link the domain cut owns no output vertex, and a component whose every
// crossing has a degenerate gradient has no centroid to divide by: both used to
// reach the output as a nonfinite coordinate or a write past the end. Neither
// is visible to an edge-degree predicate, so the ladder is walked here.
TEST_CASE("every emitted coordinate is finite", "[volume][isosurface]") {
  const auto finite_count = [](const auto &mesh) {
    const auto pts = mesh.points();
    std::size_t bad = 0;
    for (std::size_t i = 0; i < pts.size(); ++i)
      for (int d = 0; d < 3; ++d)
        bad += !std::isfinite(pts[i][d]);
    return bad;
  };
  for (const bool refine : {true, false}) {
    tf::isosurface_config config;
    config.method = tf::isosurface_method::dual_contouring;
    config.refine = refine;
    for (const int n : {24, 28, 32, 36, 40, 44, 48}) {
      INFO("n = " << n << " refine = " << refine);
      {
        INFO("slab");
        REQUIRE(finite_count(tf::make_isosurface(
                    tf::test::sampled_volume(field::slab{}, n).volume(), 0.f, config)) ==
                0);
      }
      {
        INFO("noise");
        REQUIRE(finite_count(
                    tf::make_isosurface(tf::test::sampled_volume(field::noise{}, n).volume(),
                                   0.f, config)) == 0);
      }
      {
        INFO("gyroid");
        REQUIRE(finite_count(
                    tf::make_isosurface(tf::test::sampled_volume(field::gyroid{}, n).volume(),
                                   0.f, config)) == 0);
      }
      {
        INFO("box");
        REQUIRE(finite_count(tf::make_isosurface(
                    tf::test::sampled_volume(field::box{}, n).volume(), 0.f, config)) ==
                0);
      }
    }
  }
}

// Small grids put most cells in the boundary layer, where a component's link is
// cut and the cell owns fewer vertices than it has components.
TEST_CASE("a boundary-cut link stays inside its own vertex range",
          "[volume][isosurface]") {
  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;
  for (const bool refine : {true, false}) {
    config.refine = refine;
    for (int n = 4; n <= 20; ++n) {
      INFO("n = " << n << " refine = " << refine);
      for (const auto &mesh :
           {tf::make_isosurface(tf::test::sampled_volume(field::slab{}, n).volume(), 0.f,
                           config),
            tf::make_isosurface(tf::test::sampled_volume(field::noise{}, n).volume(), 0.f,
                           config),
            tf::make_isosurface(tf::test::sampled_volume(field::gyroid{}, n).volume(), 0.f,
                           config)}) {
        const auto polys = mesh.polygons();
        const auto n_points = std::ptrdiff_t(mesh.points().size());
        for (const auto &poly : polys)
          for (int i = 0; i < int(poly.size()); ++i) {
            REQUIRE(std::ptrdiff_t(poly.indices()[i]) >= 0);
            REQUIRE(std::ptrdiff_t(poly.indices()[i]) < n_points);
          }
        const auto pts = mesh.points();
        for (std::size_t i = 0; i < pts.size(); ++i)
          for (int d = 0; d < 3; ++d)
            REQUIRE(std::isfinite(pts[i][d]));
      }
    }
  }
}

TEST_CASE("refinement and the stabilizer are the config's own fields",
          "[volume][isosurface]") {
  const auto vol = tf::test::sampled_volume(field::box{}, 40);
  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;

  config.refine = false;
  const auto plain = tf::make_isosurface(vol.volume(), 0.f, config);
  config.refine = true;
  const auto refined = tf::make_isosurface(vol.volume(), 0.f, config);
  REQUIRE(plain.polygons().size() > 0);
  REQUIRE(tf::is_closed(plain.polygons()));
  REQUIRE(tf::is_manifold(plain.polygons()));
  // refinement moves vertices; it never changes what they are connected to
  REQUIRE(plain.points().size() == refined.points().size());
  REQUIRE(plain.polygons().size() == refined.polygons().size());
  const auto &fa = plain.faces_buffer().data_buffer();
  const auto &fb = refined.faces_buffer().data_buffer();
  REQUIRE(std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(int)) == 0);
  const double truth = 8.0 * 0.62 * 0.45 * 0.33;
  REQUIRE(std::abs(double(tf::signed_volume(refined.polygons())) - truth) <
          std::abs(double(tf::signed_volume(plain.polygons())) - truth));

  // a larger pull toward the crossing centroid is still a valid surface
  for (const double stabilizer : {0.0, 0.001, 0.1, 1.0}) {
    INFO("stabilizer = " << stabilizer);
    config.stabilizer = stabilizer;
    const auto mesh = tf::make_isosurface(vol.volume(), 0.f, config);
    REQUIRE(tf::is_closed(mesh.polygons()));
    REQUIRE(tf::is_manifold(mesh.polygons()));
    const auto pts = mesh.points();
    for (std::size_t i = 0; i < pts.size(); ++i)
      for (int d = 0; d < 3; ++d)
        REQUIRE(std::isfinite(pts[i][d]));
  }
}

TEST_CASE("extraction is deterministic", "[volume][isosurface]") {
  const auto vol = tf::test::sampled_volume(field::wedge(30.f), 40);
  const auto config =
      tf::isosurface_config(tf::isosurface_method::dual_contouring);
  const auto a = tf::make_isosurface(vol.volume(), 0.f, config);
  const auto b = tf::make_isosurface(vol.volume(), 0.f, config);
  const auto &pa = a.points_buffer().data_buffer();
  const auto &pb = b.points_buffer().data_buffer();
  const auto &fa = a.faces_buffer().data_buffer();
  const auto &fb = b.faces_buffer().data_buffer();
  REQUIRE(pa.size() == pb.size());
  REQUIRE(fa.size() == fb.size());
  REQUIRE(std::memcmp(pa.data(), pb.data(), pa.size() * sizeof(float)) == 0);
  REQUIRE(std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(int)) == 0);
}

// A dual-contoured surface carries inverted pockets where a crease folds back
// on itself, so it is an operand that self-intersects. A pair graph implies
// `within` for neither side, so the call states it: the operand then enters the
// arrangement already cut against itself, and each of its cut components' sides
// is one region of space.
//
// What the booleans then owe is the solid's own algebra — a closed result, and
// what a tool keeps plus what it removes being the whole. They do not owe a
// MANIFOLD result: where a pocket's two sheets meet, the boundary the algebra
// keeps has an edge four faces share, and that is the operand's geometry, not
// the boolean's answer.
TEST_CASE("an extracted surface goes through the booleans",
          "[volume][isosurface][csg]") {
  const tf::arrangement_config resolved =
      tf::intersect_mode::primitives |
      tf::intersect_mode::resolve_crossing_contours | tf::intersect_mode::within;
  auto tool_at = [] {
    auto tool = tf::make_box_mesh<int>(0.9f, 0.9f, 3.0f);
    for (auto p : tool.points()) {
      p[0] += 0.11f;
      p[1] -= 0.07f;
    }
    return tool;
  };
  for (const int n : {32, 40, 48, 64}) {
    INFO("n = " << n);
    const auto vol = tf::test::sampled_volume(field::box{}, n);
    auto surface = tf::test::make_tagged_operand(tf::make_isosurface(
        vol.volume(), 0.f, tf::isosurface_method::dual_contouring));
    REQUIRE(tf::is_closed(surface.mesh.polygons()));
    REQUIRE(tf::is_manifold(surface.mesh.polygons()));
    // the premise this fixture's whole shape rests on: closed and manifold as
    // a mesh, and self-intersecting as a solid
    REQUIRE(tf::test::arrangement_curves_of(
                tf::test::build_self_arrangement(surface.form(), resolved))
                .size() > 0u);
    auto tool = tf::test::make_tagged_operand(tool_at());
    const double whole = double(tf::signed_volume(surface.mesh.polygons()));

    double merged = 0, parts = 0;
    for (const auto op : {tf::boolean_op::merge, tf::boolean_op::intersection,
                          tf::boolean_op::left_difference}) {
      INFO("op = " << int(op));
      auto [result, tags, faces] =
          tf::test::boolean_of(surface.form(), tool.form(), op, resolved);
      static_cast<void>(tags);
      static_cast<void>(faces);
      REQUIRE(result.polygons().size() > 0);
      REQUIRE(tf::is_closed(result.polygons()));
      const double v = double(tf::signed_volume(result.polygons()));
      if (op == tf::boolean_op::merge)
        merged = v;
      else
        parts += v;
    }
    // the tool reaches outside the surface, so their union is strictly larger
    REQUIRE(merged > whole);
    // and what the tool keeps plus what it removes is the whole
    REQUIRE(std::abs(parts - whole) < 1e-4 * std::abs(whole));
  }
}

TEST_CASE("extraction carries its index and real types",
          "[volume][isosurface]") {
  const auto vol = tf::test::sampled_volume(field::box{}, 24);
  const auto config =
      tf::isosurface_config(tf::isosurface_method::dual_contouring);
  const auto narrow = tf::make_isosurface<int, float>(vol.volume(), 0.f, config);
  const auto wide =
      tf::make_isosurface<std::int64_t, float>(vol.volume(), 0.f, config);
  REQUIRE(narrow.polygons().size() == wide.polygons().size());
  REQUIRE(tf::is_closed(wide.polygons()));
  REQUIRE(tf::is_manifold(wide.polygons()));

  // The coordinate type is the CALL's request, resolved from the samples when
  // the caller states none.
  const auto asked = tf::make_isosurface<int, double>(vol.volume(), 0.0, config);
  static_assert(
      std::is_same_v<std::decay_t<decltype(asked.points()[0][0])>, double>,
      "the request decides the output coordinate type");
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(tf::make_isosurface(vol.volume()).points()[0][0])>,
          float>,
      "an unstated request is the samples' own type");
  REQUIRE(asked.polygons().size() == narrow.polygons().size());
}

// The type a call decides in is the type it emits in, and the samples are read
// through one cast into it. A field stored wider than the request is where that
// law is load-bearing: the classifier rounds, so a crossing it states must be a
// crossing the crossing computation agrees with, or the parameter it solves is
// unbounded and the vertex leaves the grid entirely.
TEST_CASE("a narrower request decides with the samples it reads",
          "[volume][isosurface]") {
  // Two doubles on the same side of the isovalue in double, on opposite sides
  // of it once rounded to float: 0.5 - 1.5e-8 rounds below 0.5f, 0.5 - 1.4899e-8
  // rounds to 0.5f. In double their difference is 1.01e-10, so a crossing
  // parameter taken from the unrounded pair is ~148, not in [0, 1].
  const int n = 8;
  const double outside = 0.5 - 1.4899e-8;
  const double inside = 0.5 - 1.5e-8;
  tf::volume_buffer<double> vol({n, n, n}, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
  for (std::size_t i = 0; i < vol.samples_buffer().size(); ++i)
    vol.samples_buffer()[i] = outside;
  vol(3, 4, 3) = inside;

  for (const auto method : {tf::isosurface_method::flying_edges,
                            tf::isosurface_method::dual_contouring}) {
    INFO("method " << int(method));
    const auto mesh = tf::make_isosurface<int, float>(vol.volume(), 0.5f, method);
    REQUIRE(mesh.polygons().size() > 0);
    const auto pts = mesh.points();
    for (std::size_t i = 0; i < pts.size(); ++i)
      for (int d = 0; d < 3; ++d) {
        REQUIRE(std::isfinite(pts[i][d]));
        REQUIRE(pts[i][d] >= -1e-5f);
        REQUIRE(pts[i][d] <= float(n - 1) + 1e-5f);
      }
  }
}

// The same surface, over the whole {storage} x {request} matrix: the samples'
// width is a storage fact and the request is the call's, and neither may change
// which cells carry surface.
TEST_CASE("storage width and the request are independent",
          "[volume][isosurface]") {
  const auto float_vol = tf::test::sampled_volume<float>(field::box{}, 32);
  const auto double_vol = tf::test::sampled_volume<double>(field::box{}, 32);
  for (const auto method : {tf::isosurface_method::flying_edges,
                            tf::isosurface_method::dual_contouring}) {
    INFO("method " << int(method));
    const auto ff = tf::make_isosurface<int, float>(float_vol.volume(), 0.f, method);
    const auto fd = tf::make_isosurface<int, double>(float_vol.volume(), 0.0, method);
    const auto df = tf::make_isosurface<int, float>(double_vol.volume(), 0.f, method);
    const auto dd = tf::make_isosurface<int, double>(double_vol.volume(), 0.0, method);
    REQUIRE(ff.polygons().size() > 0);
    // The float field is the double field rounded, so both widths see the same
    // signs, and the request cannot move a crossing off its edge.
    REQUIRE(fd.polygons().size() == ff.polygons().size());
    REQUIRE(dd.polygons().size() == ff.polygons().size());
    REQUIRE(df.polygons().size() == ff.polygons().size());
    REQUIRE(fd.points().size() == ff.points().size());
    REQUIRE(dd.points().size() == ff.points().size());
    double worst = 0;
    for (std::size_t i = 0; i < ff.points().size(); ++i)
      for (int d = 0; d < 3; ++d)
        worst = std::max(worst, std::abs(double(ff.points()[i][d]) -
                                         dd.points()[i][d]));
    REQUIRE(worst < 1e-5);
  }
}

TEST_CASE("empty and degenerate volumes extract nothing",
          "[volume][isosurface]") {
  const auto config =
      tf::isosurface_config(tf::isosurface_method::dual_contouring);
  for (const int n : {1, 2}) {
    const auto vol = tf::test::sampled_volume(field::box{}, n);
    REQUIRE(tf::make_isosurface(vol.volume(), 0.f, config).polygons().size() == 0);
  }
  tf::volume_buffer<float> flat({8, 8, 8}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f});
  for (std::size_t i = 0; i < flat.samples_buffer().size(); ++i)
    flat.samples_buffer()[i] = 0.f;
  REQUIRE(tf::make_isosurface(flat.volume(), 0.f, config).polygons().size() == 0);
}
