/**
 * @file test_plane_refined_walls.cpp
 * @brief Tests for tf::arrangement::plane_arrangement::build_refined
 *
 * The refined walls: a stock build whole, then rounds of discovery whose
 * boundary splits are ordinary statements in the one wave and whose Steiners
 * are direct forms joining their own planes. The refined product is gated by
 * the laws it owns:
 *
 *   W1  one key per live piece — every definition of a live ticket names the
 *       same flat endpoint key, and every triangle slot naming that ticket
 *       spans exactly that key;
 *   W2  THE WATERTIGHT LAW — every carrier plane of a live piece states it: a
 *       split that reached one carrier and not another leaves the other
 *       carrying the parent, and this finds it;
 *   O   THE OWNERSHIP LAW — an unchanged group stays the world's, verbatim,
 *       for both carriers, so a plane whose ticket is -1 reads the world table,
 *       so it may name no PA-owned suffix piece;
 *   C   nothing is minted that nothing stands on — a created identity the
 *       product publishes is a triangle corner or an endpoint of a live piece.
 *
 * A refinement refusal is none of those laws: the preserve-mode producer
 * declines a plane whose recovery the stock path owns, and the product it
 * publishes there is the stock one. A refused set therefore gates nothing.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"
#include "input_lattice_for.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/static_size.hpp>
#include <trueform/arrangement/planes/make_plane_refinement_evidence.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/arrangement/planes/plane_recovery_birth.hpp>
#include <trueform/arrangement/planes/plane_recovery_statement.hpp>
#include <trueform/arrangement/planes/plane_refinement_plan.hpp>
#include <trueform/exact/dyadic_blend.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/intersect/graph/flat_of_vertex.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/polygon_intersections.hpp>
#include <trueform/topology/cdt_refine_config.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using refined_walls_index_t = tf::test::plane_index_t;
using key_t = std::array<refined_walls_index_t, 2>;
using pair_t = std::array<refined_walls_index_t, 2>;
using instance_t = std::array<refined_walls_index_t, 11>;

/// The real type each lattice width is quantized from.
template <typename Int> struct refined_walls_real_of;
template <> struct refined_walls_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct refined_walls_real_of<tf::exact::int64> {
  using type = double;
};

auto key_less(const key_t &a, const key_t &b) -> bool {
  return std::tie(a[0], a[1]) < std::tie(b[0], b[1]);
}

auto sort_unique(std::vector<key_t> &rows) -> void {
  std::sort(rows.begin(), rows.end(), key_less);
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

auto sort_unique(std::vector<refined_walls_index_t> &rows) -> void {
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

auto ordered(refined_walls_index_t a, refined_walls_index_t b) -> key_t {
  return b < a ? key_t{{b, a}} : key_t{{a, b}};
}

auto key_text(const key_t &key) -> std::string {
  return "{" + std::to_string(key[0]) + "," + std::to_string(key[1]) + "}";
}

/// Everything read off ONE arm, the way a consumer reads it: the live pieces
/// with their keys and carriers, the created identities the triangles stand
/// on, and the per-plane triangle counts. Nothing here is private state:
/// tickets, definitions and triangles are the published product.
struct product_state_t {
  std::vector<key_t> ticket_key; // per ticket, {-1,-1} while unused
  std::vector<char> live;        // per ticket: a triangle slot names it
  std::vector<pair_t> stated;    // (ticket, plane) a triangle slot states
  std::vector<pair_t> carried;   // (ticket, plane) a definition names
  std::vector<instance_t> instances;
  std::vector<key_t> live_keys;    // sorted unique
  std::vector<refined_walls_index_t> plane_tris; // per plane
  std::vector<refined_walls_index_t> created_corners;
  std::vector<refined_walls_index_t> constrained_created;
  std::vector<refined_walls_index_t> created_seen;
  std::vector<std::string> defects;
  std::size_t triangles = 0;
  std::size_t pieces = 0;
  refined_walls_index_t n_flat = 0;
  refined_walls_index_t immutable_extent = 0;

  auto interior_created() const -> std::size_t {
    return created_corners.size() - constrained_created.size();
  }
  auto orphan_created(refined_walls_index_t base_created,
                      std::size_t minted) const -> std::size_t {
    const auto begin = n_flat + base_created;
    const auto end = begin + refined_walls_index_t(minted);
    const auto first =
        std::lower_bound(created_seen.begin(), created_seen.end(), begin);
    const auto last = std::lower_bound(first, created_seen.end(), end);
    const auto seen = std::size_t(last - first);
    return minted < seen ? 0 : minted - seen;
  }
};

template <typename World, typename Product, typename VertexOffsets>
auto read_product(const World &world, const Product &product,
                  const VertexOffsets &vertex_offsets) -> product_state_t {
  product_state_t state;
  state.n_flat = product.n_flat_points();
  state.immutable_extent = product.immutable_piece_extent();
  const auto extent = product.final_piece_ticket_extent();
  const auto n_planes = product.n_planes();
  const auto triangles = product.triangles();
  const auto slots = product.slot_parents();
  state.triangles = triangles.size();
  state.ticket_key.assign(std::size_t(extent), key_t{{-1, -1}});
  state.live.assign(std::size_t(extent), char(0));

  for (refined_walls_index_t plane = 0; plane < n_planes; ++plane) {
    const auto range = product.plane_range(plane);
    state.plane_tris.push_back(range[1] - range[0]);
    for (auto triangle = range[0]; triangle < range[1]; ++triangle) {
      const auto &corners = triangles[std::size_t(triangle)];
      const auto &parents = slots[std::size_t(triangle)];
      for (std::size_t slot = 0; slot < 3; ++slot) {
        if (corners[slot] >= state.n_flat)
          state.created_corners.push_back(corners[slot]);
        const auto ticket = parents[slot];
        if (ticket == refined_walls_index_t(-1))
          continue;
        if (ticket < refined_walls_index_t(0) || ticket >= extent) {
          state.defects.push_back("slot ticket " + std::to_string(ticket) +
                                  " outside [0," + std::to_string(extent) +
                                  ") on plane " + std::to_string(plane));
          continue;
        }
        const auto key =
            ordered(corners[slot], corners[(slot + std::size_t(1)) % 3]);
        auto &known = state.ticket_key[std::size_t(ticket)];
        if (state.live[std::size_t(ticket)] == char(0)) {
          known = key;
          state.live[std::size_t(ticket)] = char(1);
        } else if (known != key)
          state.defects.push_back("W1: ticket " + std::to_string(ticket) +
                                  " spans " + key_text(known) + " and " +
                                  key_text(key));
        state.stated.push_back({ticket, plane});
      }
    }
  }

  for (refined_walls_index_t ticket = 0; ticket < extent; ++ticket) {
    if (state.live[std::size_t(ticket)] == char(0))
      continue;
    ++state.pieces;
    state.live_keys.push_back(state.ticket_key[std::size_t(ticket)]);
    for (const auto &definition : product.piece_definitions(world, ticket)) {
      const auto key = tf::arrangement::plane_recovery_flat_edge_key(
          definition, state.n_flat, vertex_offsets);
      if (key != state.ticket_key[std::size_t(ticket)])
        state.defects.push_back(
            "W1: ticket " + std::to_string(ticket) + " states " +
            key_text(state.ticket_key[std::size_t(ticket)]) +
            " but a definition names " + key_text(key));
      if (definition.face < refined_walls_index_t(0) ||
          definition.face >= world.n_faces()) {
        state.defects.push_back(
            "ticket " + std::to_string(ticket) + " names face " +
            std::to_string(definition.face) + " outside the world");
        continue;
      }
      state.carried.push_back({ticket, world.plane_of_face(definition.face)});
      const auto loop_start =
          tf::intersect::graph::plane_edge_loop_start(definition);
      const auto loop_end =
          tf::intersect::graph::plane_edge_loop_end(definition);
      state.instances.push_back(
          {key[0], key[1], world.plane_of_face(definition.face),
           definition.face, definition.object_other,
           refined_walls_index_t(definition.tag_other),
           refined_walls_index_t(definition.ordinal),
           refined_walls_index_t(definition.side),
           refined_walls_index_t(definition.flags),
           tf::intersect::graph::flat_of_vertex(
               vertex_offsets, std::int16_t(loop_start[0]), loop_start[1]),
           tf::intersect::graph::flat_of_vertex(
               vertex_offsets, std::int16_t(loop_end[0]), loop_end[1])});
    }
  }

  std::sort(state.stated.begin(), state.stated.end(), key_less);
  state.stated.erase(std::unique(state.stated.begin(), state.stated.end()),
                     state.stated.end());
  std::sort(state.carried.begin(), state.carried.end(), key_less);
  state.carried.erase(std::unique(state.carried.begin(), state.carried.end()),
                      state.carried.end());
  std::sort(state.instances.begin(), state.instances.end());
  sort_unique(state.live_keys);
  sort_unique(state.created_corners);
  for (const auto corner : state.created_corners)
    for (const auto &key : state.live_keys)
      if (key[0] == corner || key[1] == corner) {
        state.constrained_created.push_back(corner);
        break;
      }
  state.created_seen = state.created_corners;
  for (const auto &key : state.live_keys)
    for (const auto endpoint : key)
      if (endpoint >= state.n_flat)
        state.created_seen.push_back(endpoint);
  sort_unique(state.created_seen);
  return state;
}

/// THE WATERTIGHT LAW: a live piece is stated by every plane that carries a
/// definition of it. A split that reached one carrier and not another leaves
/// the other carrying the parent, and its child piece then names a plane whose
/// triangles never span the child's key.
auto watertight_defects(const product_state_t &state)
    -> std::vector<std::string> {
  std::vector<std::string> out;
  for (const auto &row : state.carried)
    if (!std::binary_search(state.stated.begin(), state.stated.end(), row,
                            key_less))
      out.push_back("W2: piece " + std::to_string(row[0]) + " " +
                    key_text(state.ticket_key[std::size_t(row[0])]) +
                    " is carried by plane " + std::to_string(row[1]) +
                    " which never states it");
  return out;
}

/// THE OWNERSHIP LAW at the published boundary: a plane still reading the
/// world table names no PA-owned suffix piece.
template <typename Product>
auto ownership_defects(const Product &product, const product_state_t &state)
    -> std::vector<std::string> {
  std::vector<std::string> out;
  const auto tickets = product.plane_tickets();
  for (const auto &row : state.stated) {
    const auto plane = row[1];
    const auto ticket = tickets.size() == 0 ? refined_walls_index_t(-1)
                                            : tickets[std::size_t(plane)];
    if (ticket == refined_walls_index_t(-1) && row[0] >= state.immutable_extent)
      out.push_back("O: plane " + std::to_string(plane) +
                    " reads the world table but names routed piece " +
                    std::to_string(row[0]));
  }
  return out;
}

template <typename Range> auto to_vector(const Range &range) {
  std::vector<refined_walls_index_t> rows;
  for (const auto value : range)
    rows.push_back(refined_walls_index_t(value));
  return rows;
}

auto check_defects(const char *arm, const std::vector<std::string> &defects)
    -> void {
  for (const auto &defect : defects)
    UNSCOPED_INFO(arm << " " << defect);
  CHECK(defects.empty());
}

/// One world, one config: the refined product gated by the laws it owns.
template <typename World, typename Subject, typename Stock,
          typename VertexOffsets>
auto check_refined_laws(const World &world, const VertexOffsets &vertex_offsets,
                        const Stock &stock, const Subject &subject,
                        refined_walls_index_t n_base_created,
                        bool require_scaled_boundary) -> void {
  const auto subject_state = read_product(world, subject, vertex_offsets);
  const auto subject_water = watertight_defects(subject_state);
  const auto subject_own = ownership_defects(subject, subject_state);

  check_defects("subject", subject_state.defects);
  check_defects("subject", subject_water);
  check_defects("subject", subject_own);

  const auto subject_created = subject.census().created;
  CHECK(subject.failed().size() == 0);
  {
    INFO("C: created identities nothing stands on");
    CHECK(subject_state.orphan_created(n_base_created, subject_created) == 0);
  }
  if (require_scaled_boundary) {
    CHECK(subject.triangles().size() == 4);
    CHECK(subject_created == 1);
    CHECK(stock.triangles().size() == 2);
  }
}

/// The refined product on a hand-built world, through the closed-world entry
/// points, against the stock product of the same world.
template <typename Int, typename GetBase, typename GetOriginal>
auto check_hand_built_wall(tf::test::plane_hand_world<Int> &fixture,
                           refined_walls_index_t n_base_created,
                           const GetBase &get_base,
                           const GetOriginal &get_original,
                           const tf::cdt_refine_config &config,
                           bool require_scaled_boundary = false) -> void {
  tf::arrangement::plane_arrangement<refined_walls_index_t, Int> stock;
  stock.record_triangle_arrangement();
  stock.build(fixture.input, n_base_created, get_base, get_original,
              fixture.vertex_offsets);
  tf::arrangement::plane_arrangement<refined_walls_index_t, Int> subject;
  subject.record_triangle_arrangement();
  subject.build_refined(fixture.input, n_base_created, get_base, get_original,
                        fixture.vertex_offsets, config);
  check_refined_laws(fixture.input, fixture.vertex_offsets, stock, subject,
                     n_base_created, require_scaled_boundary);
}

/// The config ladder every wall states for itself.
auto encroachment_enabled() -> tf::cdt_refine_config {
  tf::cdt_refine_config config;
  config.min_quality = 0.30f;
  config.split_encroached = true;
  return config;
}

auto boundaries_frozen() -> tf::cdt_refine_config {
  tf::cdt_refine_config config;
  config.min_quality = 0.30f;
  config.split_encroached = false;
  return config;
}

/// One refined build of a recovery world, gated by the subject's own laws:
/// the world's stock build states a genuine recovery, and the refined one
/// leaves no failure, no orphan and a product finer than stock.
template <typename Int>
auto check_refined_totality(tf::test::plane_hand_world<Int> fixture,
                            std::size_t required_stock_rounds) -> void {
  const auto get_original = [&](int, refined_walls_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  const auto get_base = [&](std::int16_t tag, refined_walls_index_t point) {
    return tag < 0 ? fixture.created[std::size_t(point)]
                   : get_original(int(tag), point);
  };
  const auto n_base_created = refined_walls_index_t(fixture.created.size());
  tf::arrangement::plane_arrangement<refined_walls_index_t, Int> stock;
  stock.record_triangle_arrangement();
  stock.build(fixture.input, n_base_created, get_base, get_original,
              fixture.vertex_offsets);
  tf::arrangement::plane_arrangement<refined_walls_index_t, Int> subject;
  subject.record_triangle_arrangement();
  subject.build_refined(fixture.input, n_base_created, get_base, get_original,
                        fixture.vertex_offsets, encroachment_enabled());

  const auto state =
      read_product(fixture.input, subject, fixture.vertex_offsets);
  check_defects("subject", state.defects);
  check_defects("subject", watertight_defects(state));
  check_defects("subject", ownership_defects(subject, state));

  // the totality premise: this world's stock build really did recover
  REQUIRE(stock.census().refusals != 0);
  REQUIRE(stock.census().rounds >= required_stock_rounds);
  REQUIRE(stock.failed().size() == 0);
  CHECK(subject.failed().size() == 0);

  // A world whose producer declined every plane it could have refined
  // lawfully publishes the stock product, so the finer-than-stock premise is
  // asserted only where nothing was refused.
  if (subject.refinement_refused_planes().size() == 0)
    CHECK((subject.triangles().size() != stock.triangles().size() ||
           subject.census().created != stock.census().created));
  CHECK(state.orphan_created(n_base_created, subject.census().created) == 0);
}

} // namespace

TEMPLATE_TEST_CASE("plane refined walls: the shared boundary",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;

  auto fixture = tf::test::make_plane_shared_boundary_world<Int>();
  const auto get_point = [&](int, refined_walls_index_t point) {
    return fixture.points[std::size_t(point)];
  };

  SECTION("encroachment enabled") {
    // the scaled shared boundary splits once, at its midpoint, on both planes
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(0), get_point,
                               get_point, encroachment_enabled(), true);
  }
  SECTION("boundaries frozen") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(0), get_point,
                               get_point, boundaries_frozen());
  }
}

TEMPLATE_TEST_CASE("plane refined walls: the endpoint weld",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;

  auto fixture = tf::test::make_plane_endpoint_weld_world<Int>();
  const auto get_point = [&](int, refined_walls_index_t point) {
    return fixture.points[std::size_t(point)];
  };

  SECTION("encroachment enabled") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(0), get_point,
                               get_point, encroachment_enabled());
  }
  SECTION("boundaries frozen") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(0), get_point,
                               get_point, boundaries_frozen());
  }
}

TEMPLATE_TEST_CASE("plane refined walls: the Steiner weld retry, live landings",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;

  auto fixture = tf::test::make_plane_steiner_retry_world<Int>(true);
  const auto get_original = [&](int, refined_walls_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  const auto get_base = [&](std::int16_t tag, refined_walls_index_t point) {
    return tag < 0 ? fixture.created[std::size_t(point)]
                   : get_original(int(tag), point);
  };

  SECTION("encroachment enabled") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(2), get_base,
                               get_original, encroachment_enabled());
  }
  SECTION("boundaries frozen") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(2), get_base,
                               get_original, boundaries_frozen());
  }
}

TEMPLATE_TEST_CASE("plane refined walls: the Steiner weld retry, dead landings",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;

  auto fixture = tf::test::make_plane_steiner_retry_world<Int>(false);
  const auto get_original = [&](int, refined_walls_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  const auto get_base = [&](std::int16_t tag, refined_walls_index_t point) {
    return tag < 0 ? fixture.created[std::size_t(point)]
                   : get_original(int(tag), point);
  };

  SECTION("encroachment enabled") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(2), get_base,
                               get_original, encroachment_enabled());
  }
  SECTION("boundaries frozen") {
    check_hand_built_wall<Int>(fixture, refined_walls_index_t(2), get_base,
                               get_original, boundaries_frozen());
  }
}

TEMPLATE_TEST_CASE("plane refined walls: the boundary-dominant crease",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename refined_walls_real_of<Int>::type;
  using Index = refined_walls_index_t;

  // Both products consume the SAME local-arrangement build scope: no table,
  // frame, member, orientation or merge is restated per build.
  auto config = GENERATE(encroachment_enabled(), boundaries_frozen());

  auto meshes = tf::test::make_plane_crease_meshes<Index, Real>();
  meshes.crease.points()[2][1] = Real(7) / Real(16);
  meshes.crease.points()[2][2] = Real(1) / Real(64);
  meshes.crease.points()[3][1] = Real(7) / Real(16);
  meshes.crease.points()[3][2] = -Real(1) / Real(64);

  tf::test::tagged_operand<Index, Real> crease(std::move(meshes.crease));
  tf::test::tagged_operand<Index, Real> cutter(std::move(meshes.cutter));
  using form_t = decltype(crease.form());
  std::array<form_t, 2> forms{{crease.form(), cutter.form()}};

  tf::polygon_intersections<Index, Real, Int> intersections;
  intersections.with_edge_splits(false);
  const auto intersections_lattice = tf::test::input_lattice_for(tf::make_range(forms.data(), forms.data() + forms.size()), 0.0);
  intersections.build(tf::make_range(forms.data(), forms.data() + forms.size()), intersections_lattice, tf::intersect_config{tf::intersect_mode::primitives |
                               tf::intersect_mode::resolve_crossing_contours,
                           0.0});
  const auto converter = intersections_lattice.converter();
  const auto get_original = [&, converter](int tag,
                                           Index point) -> tf::point<Int, 3> {
    return converter.convert(forms[std::size_t(tag)].points()[point]);
  };
  const auto apply_to_face = [&](int tag, Index face, const auto &apply) {
    apply(forms[std::size_t(tag)].faces()[face]);
  };
  const auto apply_to_form = [&](Index tag, const auto &apply) {
    apply(forms[std::size_t(tag)]);
  };
  tf::buffer<Index> face_offsets;
  face_offsets.allocate(forms.size() + 1);
  face_offsets[0] = 0;
  for (std::size_t tag = 0; tag < forms.size(); ++tag)
    face_offsets[tag + 1] =
        face_offsets[tag] + Index(forms[tag].faces().size());

  tf::arrangement::plane_arrangement<Index, Int> stock;
  tf::arrangement::plane_arrangement<Index, Int> subject;
  tf::intersect::graph::local_arrangement<Index, Real, Int> world;
  world.build(std::move(intersections), get_original, apply_to_face,
              apply_to_form, tf::make_range(face_offsets), true, false);
  stock.record_triangle_arrangement();
  subject.record_triangle_arrangement();
  stock.build(world, get_original);
  subject.build_refined(world, get_original, apply_to_form, config);

  check_refined_laws(world, world.vertex_offsets(), stock, subject,
                     refined_walls_index_t(world.n_created_points()), false);
}

TEMPLATE_TEST_CASE("plane refined walls: recovery totality, uncut neighbor",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  check_refined_totality<TestType>(
      tf::test::make_plane_uncut_neighbor_world<TestType>(), std::size_t(2));
}

TEMPLATE_TEST_CASE("plane refined walls: recovery totality, wave-born",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  check_refined_totality<TestType>(
      tf::test::make_plane_bent_child_world<TestType>(false), std::size_t(3));
}

TEMPLATE_TEST_CASE("plane refined walls: refinement lands on both tiers",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;

  // The stock build leaves this world genuinely mixed — the bent child's two
  // carriers local, the slender bystander still reading the world table. The
  // gate is the subject's law set plus the evidence that refinement reached
  // BOTH tiers: a plane the stock build left at ticket -1 is local afterwards
  // (an immutable-tier split landed), and a plane the stock build already
  // owned carries strictly more triangles (a current-tier plane refined
  // again).
  auto fixture = tf::test::make_plane_mixed_tier_world<Int>();
  const auto get_original = [&](int, refined_walls_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  const auto get_base = [&](std::int16_t tag, refined_walls_index_t point) {
    return tag < 0 ? fixture.points[std::size_t(point)]
                   : get_original(int(tag), point);
  };

  tf::arrangement::plane_arrangement<refined_walls_index_t, Int> stock;
  stock.record_triangle_arrangement();
  stock.build(fixture.input, refined_walls_index_t(0), get_base, get_original,
              fixture.vertex_offsets);
  const auto stock_tickets = to_vector(stock.plane_tickets());
  std::vector<refined_walls_index_t> stock_tris;
  for (refined_walls_index_t plane = 0; plane < stock.n_planes(); ++plane) {
    const auto range = stock.plane_range(plane);
    stock_tris.push_back(range[1] - range[0]);
  }
  std::size_t stock_local = 0;
  for (const auto ticket : stock_tickets)
    stock_local += std::size_t(ticket != refined_walls_index_t(-1));
  REQUIRE(stock_tickets.size() != 0);
  REQUIRE(stock_local != 0);
  REQUIRE(stock_local != stock_tickets.size());

  const bool split_encroached = GENERATE(true, false);
  auto config = split_encroached ? encroachment_enabled() : boundaries_frozen();
  tf::arrangement::plane_arrangement<refined_walls_index_t, Int> subject;
  subject.record_triangle_arrangement();
  subject.build_refined(fixture.input, refined_walls_index_t(0), get_base,
                        get_original, fixture.vertex_offsets, config);

  const auto state =
      read_product(fixture.input, subject, fixture.vertex_offsets);
  check_defects("subject", state.defects);
  check_defects("subject", watertight_defects(state));
  check_defects("subject", ownership_defects(subject, state));
  CHECK(subject.failed().size() == 0);

  const auto tickets = to_vector(subject.plane_tickets());
  std::size_t ported = 0;
  std::size_t deepened = 0;
  for (std::size_t plane = 0; plane < stock_tickets.size(); ++plane) {
    const auto after =
        tickets.size() == 0 ? refined_walls_index_t(-1) : tickets[plane];
    if (stock_tickets[plane] == refined_walls_index_t(-1) &&
        after != refined_walls_index_t(-1))
      ++ported;
    if (stock_tickets[plane] != refined_walls_index_t(-1) &&
        std::size_t(plane) < state.plane_tris.size() &&
        state.plane_tris[plane] > stock_tris[plane])
      ++deepened;
  }
  // frozen boundaries evidence no landing at all, which is the config's own
  // statement, not a defect
  if (split_encroached) {
    INFO("ported " << ported << " deepened " << deepened);
    CHECK(ported != 0);
    CHECK(deepened != 0);
  }
  CHECK(state.orphan_created(refined_walls_index_t(0),
                             subject.census().created) == 0);
}

TEMPLATE_TEST_CASE("plane refined walls: a mixed-endpoint name mints in place",
                   "[cut][planes][refined]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;

  // A definition with one created and one original endpoint keys created-FIRST,
  // as tag -1, and flattens created-LAST, past every original vertex: its own
  // order and its NAME's are OPPOSITE. This world states one such definition,
  // and the case states two refinement splits on it at 1/4 and 1/2 of the
  // definition — the asymmetric set a mirror cannot disguise.
  //
  // THE LAW is the minted POSITION: the identity a wave mints is blended from
  // the name's endpoints at the name's parameter, so that blend must equal the
  // point the producer chose on the definition. A parameter bound to the name
  // in the definition's frame blends 1/4 of the way from the WRONG end.
  const auto fixture = tf::test::make_plane_multi_parent_world<Int>();
  const auto &tables = fixture.input.tables();
  const auto n_flat = fixture.vertex_offsets[fixture.vertex_offsets.size() - 1];
  const auto whole = param_t(1) << tf::exact::meta<Int>::param_bits;
  const auto quarter = whole >> 2;
  const auto half = whole >> 1;
  const auto point_of = [&](std::int16_t tag, refined_walls_index_t id) {
    return tag < 0 ? fixture.created[std::size_t(id)]
                   : fixture.points[std::size_t(id)];
  };
  const auto flat_point = [&](refined_walls_index_t flat) {
    return flat >= n_flat ? fixture.created[std::size_t(flat - n_flat)]
                          : fixture.points[std::size_t(flat)];
  };
  const auto blend = [](const tf::point<Int, 3> &lo,
                        const tf::point<Int, 3> &hi, param_t parameter) {
    tf::point<Int, 3> out;
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      out[coordinate] = tf::exact::dyadic_blend<Int>(lo[coordinate],
                                                     hi[coordinate], parameter);
    return out;
  };

  refined_walls_index_t inverted_group = refined_walls_index_t(-1);
  for (refined_walls_index_t group = 0; group < tables.n_canon(); ++group) {
    const auto ends = tf::arrangement::plane_recovery_flat_edge_ends(
        tables.canon_group(group)[0], n_flat, fixture.vertex_offsets);
    if (tf::arrangement::plane_recovery_flat_edge_inverted(ends))
      inverted_group = group;
  }
  REQUIRE(inverted_group != refined_walls_index_t(-1));

  const auto &def = tables.canon_group(inverted_group)[0];
  const auto ends = tf::arrangement::plane_recovery_flat_edge_ends(
      def, n_flat, fixture.vertex_offsets);
  const std::array<refined_walls_index_t, 2> feature{{ends[1], ends[0]}};

  tf::buffer<
      tf::arrangement::plane_refinement_split<refined_walls_index_t, Int>>
      splits;
  for (const auto parameter : {quarter, half})
    splits.push_back({feature, def.face, inverted_group, parameter,
                      tf::arrangement::plane_refinement_immutable_source,
                      std::uint8_t(1)});

  tf::buffer<
      tf::arrangement::plane_recovery_statement<refined_walls_index_t, param_t>>
      statements;
  tf::buffer<
      tf::arrangement::plane_recovery_proposal<refined_walls_index_t, param_t>>
      topology;
  tf::arrangement::make_plane_refinement_evidence(
      splits, refined_walls_index_t(6), whole, true, statements, topology);

  for (std::size_t row = 0; row < splits.size(); ++row) {
    INFO("split " << row);
    // the name's blend IS the producer's point on the definition
    CHECK(blend(flat_point(feature[0]), flat_point(feature[1]),
                statements[row].name.parameter) ==
          blend(point_of(def.point_tag_0, def.point_0),
                point_of(def.point_tag_1, def.point_1), splits[row].parameter));
    CHECK(statements[row].name.parameter == topology[row].parameter);
  }

  // the chain the respan builds runs along the definition, `point_0` first
  tf::buffer<refined_walls_index_t> class_offsets;
  const auto n_classes =
      tf::arrangement::close_plane_recovery_proposals(topology, class_offsets);
  tf::buffer<refined_walls_index_t> class_id;
  for (refined_walls_index_t cls = 0; cls < n_classes; ++cls)
    class_id.push_back(n_flat + refined_walls_index_t(cls));
  const tf::buffer<std::array<refined_walls_index_t, 3>> no_merges;
  tf::buffer<refined_walls_index_t> split_edge, split_tier, split_offsets,
      split_data;
  std::size_t out_of_span = 0;
  std::size_t on_endpoint = 0;
  tf::arrangement::order_plane_recovery_splits(
      tables, tables, topology, class_offsets, class_id, no_merges,
      fixture.vertex_offsets, n_classes, whole, true, split_edge, split_tier,
      split_offsets, split_data, out_of_span, on_endpoint);
  REQUIRE(split_edge.size() == 1);
  REQUIRE(split_data.size() == 2);
  CHECK(out_of_span == 0);
  CHECK(on_endpoint == 0);

  // every cut speaks its class's name; on this definition the name's frame is
  // the mirror, so the chain ascends exactly when the mirrored run does
  std::vector<param_t> chain;
  for (const auto cut : split_data) {
    const auto cls = std::size_t(cut - n_flat);
    chain.push_back(
        param_t(whole - topology[std::size_t(class_offsets[cls])].parameter));
  }
  CHECK(chain[0] < chain[1]);
}
