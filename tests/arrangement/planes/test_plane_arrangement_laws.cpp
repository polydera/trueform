/**
 * @file test_plane_arrangement_laws.cpp
 * @brief The published laws of tf::arrangement::plane_arrangement, on synthetic
 * scenes
 *
 * Every law here is read from the arrangement's PUBLISHED surface only:
 *
 *   L   THE LIVE-SET LAW — the pieces the planes that BOUND AREA name are
 *       exactly the pieces the emitted slots name. A wave that subdivides a
 *       piece moves its carriers to the children and leaves the parent
 *       standing in the ticket address space, so the address space is not the
 *       live set.
 *   W1  one key per live piece — every definition of a live ticket names the
 *       same flat endpoint key, and every slot naming it spans that key.
 *   W2  THE WATERTIGHT LAW — every carrier plane that bounds area states the
 *       live pieces it carries.
 *   O   THE OWNERSHIP LAW — a plane whose ticket is -1 reads the world table
 *       and may name no PA-owned suffix piece. A group whose statement changes
 *       goes local and takes every carrier of it in the same wave, and AN
 *       UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH CARRIERS — so a
 *       carrier still reading the world names only the immutable prefix, and
 *       the ticket it publishes is the ticket its ported peer publishes.
 *   I   THE CUT INCIDENCE LAW — for a live cut piece on a carrier plane that
 *       bounds area, the emitted slot count is 2 when every instance on that
 *       plane is an interior cut and 1 when a base-loop instance shares it.
 *
 * The instrument is proven able to fail: hiding one published piece must break
 * L, and dropping one emitted cut slot must break I.
 *
 * The same scenes carry the two identity gates: one scene built twice
 * publishes the same product, and a refinement configured to do nothing
 * publishes the stock product.
 *
 * A LINE CARRIER is one more carrier of that same tier. It bounds no area, so
 * its triangulation is empty, and it states the landings and welds its
 * constraint set holds like any other — including on a line parallel to the
 * axis its own frame drops, where a projection that was not its own would
 * merge distinct lattice points into one identity.
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
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/arrangement/planes/plane_recovery_birth.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/polygon_intersections.hpp>
#include <trueform/topology/cdt_refine_config.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using laws_index_t = tf::test::plane_index_t;
using key_t = std::array<laws_index_t, 2>;

/// The real type each lattice width is quantized from.
template <typename Int> struct laws_real_of;
template <> struct laws_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct laws_real_of<tf::exact::int64> {
  using type = double;
};

auto ordered(laws_index_t a, laws_index_t b) -> key_t {
  return b < a ? key_t{{b, a}} : key_t{{a, b}};
}

struct oracle_report_t {
  std::size_t planes = 0;
  std::size_t immutable_planes = 0;
  std::size_t local_planes = 0;
  std::size_t triangles = 0;
  laws_index_t immutable_extent = 0;
  laws_index_t ticket_extent = 0;
  std::size_t published = 0;
  std::size_t emitted = 0;
  std::size_t published_only = 0;
  std::size_t emitted_only = 0;
  std::size_t cut_pieces = 0;
  std::size_t cut_occurrences = 0;
  std::size_t cut_expected_rows = 0;
  std::size_t cut_plane_defects = 0;
  std::size_t w1_defects = 0;
  std::size_t w2_defects = 0;
  std::size_t ownership_defects = 0;
  std::vector<std::string> samples;
};

/// One (ticket, plane) incidence, observed or required.
struct incidence_t {
  laws_index_t ticket;
  laws_index_t plane;
  std::size_t multiplicity;

  auto operator<(const incidence_t &other) const -> bool {
    return ticket != other.ticket ? ticket < other.ticket : plane < other.plane;
  }
};

auto collapse(std::vector<incidence_t> &rows, bool take_minimum) -> void {
  std::sort(rows.begin(), rows.end());
  std::size_t kept = 0;
  for (const auto &row : rows) {
    if (kept != 0 && rows[kept - 1].ticket == row.ticket &&
        rows[kept - 1].plane == row.plane) {
      rows[kept - 1].multiplicity =
          take_minimum ? std::min(rows[kept - 1].multiplicity, row.multiplicity)
                       : rows[kept - 1].multiplicity + row.multiplicity;
      continue;
    }
    rows[kept++] = row;
  }
  rows.resize(kept);
}

/// The counterfactuals that must turn this instrument red: `drop_occurrence`
/// hides one emitted cut slot, `hide_publication` takes one live piece out of
/// the plane tier's statement.
enum class mutation_t { none, drop_occurrence, hide_publication };

template <typename World, typename VertexOffsets, typename Product>
auto evaluate(const World &world, const VertexOffsets &vertex_offsets,
              const Product &product, mutation_t mutation = mutation_t::none)
    -> oracle_report_t {
  oracle_report_t out;
  const auto n_flat = product.n_flat_points();
  const auto immutable_extent = product.immutable_piece_extent();
  const auto ticket_extent = product.final_piece_ticket_extent();
  const auto n_planes = product.n_planes();
  const auto triangles = product.triangles();
  const auto slot_parents = product.slot_parents();
  const auto plane_blocks = product.plane_tickets();
  out.planes = std::size_t(n_planes);
  out.triangles = triangles.size();
  out.immutable_extent = immutable_extent;
  out.ticket_extent = ticket_extent;

  const auto sample = [&out](std::string row) {
    if (out.samples.size() < 12)
      out.samples.push_back(std::move(row));
  };
  // L, W2 and I are laws of the carriers that BOUND AREA: a line carrier
  // states its pieces and emits no triangle to name them with. A plane the
  // refinement promoted is not the world's, and bounds area by construction.
  const auto bounds_area = [&world](laws_index_t plane) {
    if (plane >= world.n_planes())
      return true;
    const auto &normal = world.frame(plane).plane_n;
    using normal_t = std::decay_t<decltype(normal[0])>;
    return !(normal[0] == normal_t(0) && normal[1] == normal_t(0) &&
             normal[2] == normal_t(0));
  };

  // L: what the plane tier names.
  std::vector<unsigned char> published(std::size_t(ticket_extent), 0);
  for (laws_index_t plane = 0; plane < n_planes; ++plane) {
    const auto block = plane_blocks.size() == 0
                           ? laws_index_t(-1)
                           : plane_blocks[std::size_t(plane)];
    const bool area = bounds_area(plane);
    if (block == laws_index_t(-1)) {
      ++out.immutable_planes;
      if (area)
        for (const auto row : world.plane_edges(plane))
          published[std::size_t(world.edge_defs()[std::size_t(row)].id)] = 1;
      continue;
    }
    ++out.local_planes;
    if (area)
      for (const auto ticket : product.current_plane_tickets(world, block))
        published[std::size_t(ticket)] = 1;
  }
  if (mutation == mutation_t::hide_publication)
    for (laws_index_t ticket = 0; ticket < ticket_extent; ++ticket)
      if (published[std::size_t(ticket)] != 0) {
        published[std::size_t(ticket)] = 0;
        break;
      }

  // W1 and the observed incidences.
  std::vector<unsigned char> emitted(std::size_t(ticket_extent), 0);
  std::vector<key_t> ticket_key(std::size_t(ticket_extent), key_t{{-1, -1}});
  std::vector<incidence_t> observed;
  observed.reserve(triangles.size());
  bool dropped_one = false;
  for (laws_index_t plane = 0; plane < n_planes; ++plane) {
    const auto range = product.plane_range(plane);
    for (auto row = range[0]; row < range[1]; ++row)
      for (std::size_t slot = 0; slot < 3; ++slot) {
        const auto ticket = slot_parents[std::size_t(row)][slot];
        if (ticket == laws_index_t(-1))
          continue;
        const auto key =
            ordered(triangles[std::size_t(row)][slot],
                    triangles[std::size_t(row)][(slot + std::size_t(1)) % 3]);
        if (emitted[std::size_t(ticket)] == 0) {
          emitted[std::size_t(ticket)] = 1;
          ticket_key[std::size_t(ticket)] = key;
        } else if (ticket_key[std::size_t(ticket)] != key) {
          ++out.w1_defects;
          sample("W1 ticket " + std::to_string(ticket) + " spans {" +
                 std::to_string(ticket_key[std::size_t(ticket)][0]) + "," +
                 std::to_string(ticket_key[std::size_t(ticket)][1]) +
                 "} and {" + std::to_string(key[0]) + "," +
                 std::to_string(key[1]) + "}");
        }
        if (mutation == mutation_t::drop_occurrence && !dropped_one) {
          bool cut = false;
          for (const auto &definition :
               product.piece_definitions(world, ticket))
            cut = cut || definition.ordinal < 0;
          if (cut) {
            dropped_one = true;
            continue;
          }
        }
        observed.push_back({ticket, plane, 1});
      }
  }
  collapse(observed, false);

  for (laws_index_t ticket = 0; ticket < ticket_extent; ++ticket) {
    const auto is_published = published[std::size_t(ticket)] != 0;
    const auto is_emitted = emitted[std::size_t(ticket)] != 0;
    out.published += std::size_t(is_published);
    out.emitted += std::size_t(is_emitted);
    if (is_published && !is_emitted) {
      ++out.published_only;
      sample("L ticket " + std::to_string(ticket) +
             " is named by a plane block and by no triangle");
    }
    if (is_emitted && !is_published) {
      ++out.emitted_only;
      sample("L ticket " + std::to_string(ticket) +
             " is named by a triangle and by no plane block");
    }
  }

  // W2, O and the required cut incidences.
  std::vector<incidence_t> required;
  for (laws_index_t ticket = 0; ticket < ticket_extent; ++ticket) {
    if (emitted[std::size_t(ticket)] == 0)
      continue;
    bool cut = false;
    for (const auto &definition : product.piece_definitions(world, ticket))
      cut = cut || definition.ordinal < 0;
    out.cut_pieces += std::size_t(cut);
    for (const auto &definition : product.piece_definitions(world, ticket)) {
      const auto key = tf::arrangement::plane_recovery_flat_edge_key(
          definition, n_flat, vertex_offsets);
      if (definition.id != ticket || key != ticket_key[std::size_t(ticket)]) {
        ++out.w1_defects;
        sample("W1 ticket " + std::to_string(ticket) + " spans {" +
               std::to_string(ticket_key[std::size_t(ticket)][0]) + "," +
               std::to_string(ticket_key[std::size_t(ticket)][1]) +
               "} but a definition of id " + std::to_string(definition.id) +
               " names {" + std::to_string(key[0]) + "," +
               std::to_string(key[1]) + "}");
        continue;
      }
      const auto plane =
          definition.face < world.n_faces()
              ? world.plane_of_face(definition.face)
              : world.n_planes() + (definition.face - world.n_faces());
      const auto carried = std::lower_bound(observed.begin(), observed.end(),
                                            incidence_t{ticket, plane, 0});
      if (bounds_area(plane) &&
          (carried == observed.end() || carried->ticket != ticket ||
           carried->plane != plane)) {
        ++out.w2_defects;
        sample("W2 piece " + std::to_string(ticket) + " is carried by plane " +
               std::to_string(plane) + " which never states it");
        continue;
      }
      const auto block = plane_blocks.size() == 0
                             ? laws_index_t(-1)
                             : plane_blocks[std::size_t(plane)];
      if (block == laws_index_t(-1) && ticket >= immutable_extent) {
        ++out.ownership_defects;
        sample("O plane " + std::to_string(plane) +
               " reads the world table but names routed piece " +
               std::to_string(ticket));
      }
      if (cut && bounds_area(plane))
        required.push_back(
            {ticket, plane,
             definition.ordinal < 0 ? std::size_t(2) : std::size_t(1)});
    }
  }
  collapse(required, true);
  out.cut_expected_rows = required.size();

  for (const auto &row : required) {
    const auto seen = std::lower_bound(observed.begin(), observed.end(), row);
    const auto multiplicity = seen != observed.end() &&
                                      seen->ticket == row.ticket &&
                                      seen->plane == row.plane
                                  ? seen->multiplicity
                                  : std::size_t(0);
    out.cut_occurrences += multiplicity;
    if (multiplicity == row.multiplicity)
      continue;
    ++out.cut_plane_defects;
    sample("I cut piece " + std::to_string(row.ticket) + " {" +
           std::to_string(ticket_key[std::size_t(row.ticket)][0]) + "," +
           std::to_string(ticket_key[std::size_t(row.ticket)][1]) +
           "} on plane " + std::to_string(row.plane) + " multiplicity " +
           std::to_string(multiplicity) + "/" +
           std::to_string(row.multiplicity));
  }
  return out;
}

auto check_laws(const oracle_report_t &report) -> void {
  for (const auto &row : report.samples)
    UNSCOPED_INFO(row);
  CHECK(report.published_only == 0);
  CHECK(report.emitted_only == 0);
  CHECK(report.w1_defects == 0);
  CHECK(report.w2_defects == 0);
  CHECK(report.ownership_defects == 0);
  CHECK(report.cut_plane_defects == 0);
}

/// One triangle of the product, in the shape a consumer reads it: the plane it
/// belongs to, its corners, the piece each slot publishes, and the stack state
/// the emitter recorded.
using record_t = std::array<laws_index_t, 9>;

template <typename Product>
auto records_of(const Product &product) -> std::vector<record_t> {
  std::vector<record_t> records;
  const auto triangles = product.triangles();
  const auto slots = product.slot_parents();
  for (laws_index_t plane = 0; plane < product.n_planes(); ++plane) {
    const auto range = product.plane_range(plane);
    for (auto row = range[0]; row < range[1]; ++row)
      records.push_back(
          {plane, triangles[std::size_t(row)][0],
           triangles[std::size_t(row)][1], triangles[std::size_t(row)][2],
           slots[std::size_t(row)][0], slots[std::size_t(row)][1],
           slots[std::size_t(row)][2],
           laws_index_t(product.stacked()[std::size_t(row)]),
           product.coplanar_of()[std::size_t(row)] == laws_index_t(-1)
               ? laws_index_t(0)
               : laws_index_t(1)});
  }
  std::sort(records.begin(), records.end());
  return records;
}

/// Two boxes overlapping in one corner: every operand face that the other box
/// crosses carries interior cuts, so the scene states cuts, and the shared
/// corner region states them on several planes at once.
template <typename Real> auto make_scene_meshes() {
  const auto translated = [](tf::polygons_buffer<laws_index_t, Real, 3, 3> mesh,
                             Real dx, Real dy, Real dz) {
    for (auto &&point : mesh.points_buffer()) {
      point[0] += dx;
      point[1] += dy;
      point[2] += dz;
    }
    return mesh;
  };
  std::array<tf::polygons_buffer<laws_index_t, Real, 3, 3>, 2> meshes{
      {tf::make_box_mesh<laws_index_t, Real>(Real(2), Real(2), Real(2)),
       translated(
           tf::make_box_mesh<laws_index_t, Real>(Real(2), Real(2), Real(2)),
           Real(1), Real(1), Real(1))}};
  return meshes;
}

/// One arm of the scene: the product's records and the three evaluations of
/// it — the laws, and the two mutations that must break them.
struct arm_state_t {
  oracle_report_t report;
  oracle_report_t dropped;
  oracle_report_t hidden;
  std::vector<record_t> records;
  std::size_t failed = 0;
  std::size_t triangles = 0;
  std::size_t created = 0;
};

/// The three arms one scene answers: the stock product, the refined one, and
/// the one a refinement configured to split nothing publishes.
struct scene_state_t {
  arm_state_t stock;
  arm_state_t refined;
  std::vector<record_t> null_refined_records;
};

/// Build the scene's local arrangement once and read every arm off it.
template <typename Int, typename Real> auto build_scene() -> scene_state_t {
  auto meshes = make_scene_meshes<Real>();
  tf::test::tagged_operand<laws_index_t, Real> first(std::move(meshes[0]));
  tf::test::tagged_operand<laws_index_t, Real> second(std::move(meshes[1]));
  using form_t = decltype(first.form());
  std::array<form_t, 2> forms{{first.form(), second.form()}};

  tf::polygon_intersections<laws_index_t, Real, Int> intersections;
  intersections.with_edge_splits(false);
  const auto intersections_lattice = tf::test::input_lattice_for(tf::make_range(forms.data(), forms.data() + forms.size()), 0.0);
  intersections.build(tf::make_range(forms.data(), forms.data() + forms.size()), intersections_lattice, tf::intersect_config{tf::intersect_mode::primitives |
                               tf::intersect_mode::resolve_crossing_contours,
                           0.0});
  const auto converter = intersections_lattice.converter();
  const auto get_original =
      [&, converter](int tag, laws_index_t point) -> tf::point<Int, 3> {
    return converter.convert(forms[std::size_t(tag)].points()[point]);
  };
  const auto apply_to_face = [&](int tag, laws_index_t face,
                                 const auto &apply) {
    apply(forms[std::size_t(tag)].faces()[face]);
  };
  const auto apply_to_form = [&](laws_index_t tag, const auto &apply) {
    apply(forms[std::size_t(tag)]);
  };
  tf::buffer<laws_index_t> face_offsets;
  face_offsets.allocate(forms.size() + 1);
  face_offsets[0] = 0;
  for (std::size_t tag = 0; tag < forms.size(); ++tag)
    face_offsets[tag + 1] =
        face_offsets[tag] + laws_index_t(forms[tag].faces().size());

  tf::cdt_refine_config quality;
  quality.min_quality = 0.3f;
  quality.split_encroached = true;
  tf::cdt_refine_config nothing;
  nothing.min_quality = 0.0f;
  nothing.split_encroached = false;
  tf::arrangement::plane_arrangement<laws_index_t, Int> stock;
  tf::arrangement::plane_arrangement<laws_index_t, Int> refined;
  tf::arrangement::plane_arrangement<laws_index_t, Int> null_refined;
  tf::intersect::graph::local_arrangement<laws_index_t, Real, Int> world;
  world.build(std::move(intersections), get_original, apply_to_face,
              apply_to_form, tf::make_range(face_offsets), false, false);
  // the piece each slot publishes is what these laws read
  stock.record_triangle_arrangement();
  refined.record_triangle_arrangement();
  null_refined.record_triangle_arrangement();
  stock.build(world, get_original);
  refined.build_refined(world, get_original, apply_to_form, quality);
  null_refined.build_refined(world, get_original, apply_to_form, nothing);

  const auto read_arm = [&](const auto &product) {
    arm_state_t arm;
    arm.report = evaluate(world, world.vertex_offsets(), product);
    arm.dropped = evaluate(world, world.vertex_offsets(), product,
                           mutation_t::drop_occurrence);
    arm.hidden = evaluate(world, world.vertex_offsets(), product,
                          mutation_t::hide_publication);
    arm.records = records_of(product);
    arm.failed = product.failed().size();
    arm.triangles = product.triangles().size();
    arm.created = product.census().created;
    return arm;
  };
  scene_state_t state;
  state.stock = read_arm(stock);
  state.refined = read_arm(refined);
  state.null_refined_records = records_of(null_refined);
  return state;
}

/// One hand-built world through both entry points. These worlds recover, so
/// their planes reach the local tier — the branch of the live-set law a clean
/// scene never takes.
template <typename Int>
auto check_hand_built_laws(tf::test::plane_hand_world<Int> fixture,
                           bool refined) -> oracle_report_t {
  const auto get_original = [&](int, laws_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  const auto get_base = [&](std::int16_t tag, laws_index_t point) {
    return tag < 0 && fixture.created.size() != 0
               ? fixture.created[std::size_t(point)]
               : get_original(int(tag), point);
  };
  tf::cdt_refine_config config;
  config.min_quality = 0.3f;
  config.split_encroached = true;
  tf::arrangement::plane_arrangement<laws_index_t, Int> product;
  product.record_triangle_arrangement();
  if (refined)
    product.build_refined(fixture.input, laws_index_t(fixture.created.size()),
                          get_base, get_original, fixture.vertex_offsets,
                          config);
  else
    product.build(fixture.input, laws_index_t(fixture.created.size()), get_base,
                  get_original, fixture.vertex_offsets);
  CHECK(product.failed().size() == 0);
  return evaluate(fixture.input, fixture.vertex_offsets, product);
}

/// The distinct flat identities the live definitions of one plane name.
template <typename World, typename VertexOffsets, typename Product>
auto carrier_identities(const World &world, const VertexOffsets &vertex_offsets,
                        const Product &product, laws_index_t plane)
    -> std::vector<laws_index_t> {
  std::vector<laws_index_t> ids;
  const auto n_flat = product.n_flat_points();
  const auto blocks = product.plane_tickets();
  const auto block =
      blocks.size() == 0 ? laws_index_t(-1) : blocks[std::size_t(plane)];
  const auto take = [&](const auto &definition) {
    const auto key = tf::arrangement::plane_recovery_flat_edge_key(
        definition, n_flat, vertex_offsets);
    ids.push_back(key[0]);
    ids.push_back(key[1]);
  };
  if (block == laws_index_t(-1))
    for (const auto row : world.plane_edges(plane))
      take(world.edge_defs()[std::size_t(row)]);
  else
    for (const auto &definition : product.current_plane_defs(world, block))
      take(definition);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

} // namespace

TEMPLATE_TEST_CASE("plane arrangement: a line carrier states its facts",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;

  const bool refined = GENERATE(false, true);
  const int axis = GENERATE(0, 1, 2);
  const bool twin = GENERATE(false, true);
  INFO("refined build: " << refined << " axis: " << axis << " twin: " << twin);

  auto fixture = tf::test::make_plane_line_carrier_world<Int>(axis, twin);
  const auto get_original = [&](int, laws_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  tf::cdt_refine_config config;
  config.min_quality = 0.3f;
  config.split_encroached = true;
  tf::arrangement::plane_arrangement<laws_index_t, Int> product;
  product.record_triangle_arrangement();
  if (refined)
    product.build_refined(fixture.input, laws_index_t(0), get_original,
                          get_original, fixture.vertex_offsets, config);
  else
    product.build(fixture.input, laws_index_t(0), get_original, get_original,
                  fixture.vertex_offsets);

  // the line carrier bounds no area, so it emits nothing — and it refuses
  // its way to the landing that splits the span it shares with the area
  // plane, whose triangle therefore becomes two
  CHECK(product.failed().size() == 0);
  const auto area = product.plane_range(0);
  const auto line = product.plane_range(1);
  CHECK(line[1] - line[0] == 0);
  REQUIRE(area[1] - area[0] == 2);

  // the two triangles of the area plane meet along the landed identity
  const auto triangles = product.triangles();
  std::vector<laws_index_t> shared;
  for (std::size_t slot = 0; slot < 3; ++slot)
    for (std::size_t other = 0; other < 3; ++other)
      if (triangles[std::size_t(area[0])][slot] ==
          triangles[std::size_t(area[0] + 1)][other])
        shared.push_back(triangles[std::size_t(area[0])][slot]);
  std::sort(shared.begin(), shared.end());
  shared.erase(std::unique(shared.begin(), shared.end()), shared.end());
  CHECK(shared.size() == 2);

  // THE WELD TRUTHFULNESS GATE: the projection of a line carrier is
  // injective on it, so its four distinct lattice points stay four
  // identities whatever the line's direction — and the twin's fifth name,
  // which IS one of those positions, is the one weld the carrier states.
  CHECK(carrier_identities(fixture.input, fixture.vertex_offsets, product,
                           laws_index_t(1))
            .size() == 4);

  check_laws(evaluate(fixture.input, fixture.vertex_offsets, product));
}

TEMPLATE_TEST_CASE("plane arrangement: a stated split is not stated twice",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;

  const int axis = GENERATE(0, 1, 2);
  INFO("axis: " << axis);

  // THE INPUT IS THE AUTHORITY. A fact already applied to the tables is not
  // a fact this tier can find: the identity that WAS strictly inside the
  // shared span arrives as the endpoint of two abutting pieces, and an
  // endpoint is no crossing. The same world unsplit is the counterfactual —
  // there the tier states the landing itself, in a second round.
  auto stated = tf::test::make_plane_line_carrier_world<Int>(
      axis, /*twin=*/false, /*presplit=*/true);
  auto unstated = tf::test::make_plane_line_carrier_world<Int>(axis);
  const auto build = [&](tf::test::plane_hand_world<Int> &fixture) {
    const auto get_original = [&](int, laws_index_t point) {
      return fixture.points[std::size_t(point)];
    };
    tf::arrangement::plane_arrangement<laws_index_t, Int> product;
    product.record_triangle_arrangement();
    product.build(fixture.input, laws_index_t(0), get_original, get_original,
                  fixture.vertex_offsets);
    return product;
  };

  const auto answered = build(stated);
  const auto &census = answered.census();
  CHECK(census.rounds == 1);
  CHECK(census.refusals == 0);
  CHECK(census.landings == 0);
  CHECK(census.splits == 0);
  CHECK(census.created == 0);
  CHECK(answered.failed().size() == 0);
  const auto area = answered.plane_range(0);
  const auto line = answered.plane_range(1);
  CHECK(area[1] - area[0] == 2);
  CHECK(line[1] - line[0] == 0);
  // the pieces carry the ORIGINAL names, which is what stating it upstream
  // buys: no corner of the shared span is a created identity
  const auto triangles = answered.triangles();
  for (auto row = area[0]; row < area[1]; ++row)
    for (std::size_t corner = 0; corner < 3; ++corner)
      CHECK(triangles[std::size_t(row)][corner] < answered.n_flat_points());

  const auto found = build(unstated);
  CHECK(found.census().rounds != 1);
  CHECK(found.census().refusals != 0);
  CHECK(found.census().splits != 0);
}

TEMPLATE_TEST_CASE("plane arrangement laws: the recovery worlds",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;

  const bool refined = GENERATE(false, true);
  INFO("refined build: " << refined);

  SECTION("the bent child") {
    const auto report = check_hand_built_laws(
        tf::test::make_plane_bent_child_world<Int>(false), refined);
    REQUIRE(report.local_planes != 0);
    check_laws(report);
  }
  SECTION("the uncut neighbor") {
    const auto report = check_hand_built_laws(
        tf::test::make_plane_uncut_neighbor_world<Int>(), refined);
    REQUIRE(report.local_planes != 0);
    check_laws(report);
  }
  SECTION("the mixed tier") {
    const auto report = check_hand_built_laws(
        tf::test::make_plane_mixed_tier_world<Int>(), refined);
    REQUIRE(report.local_planes != 0);
    // the stock build leaves the bystander reading the world table; a refined
    // round is free to route it too
    if (!refined)
      REQUIRE(report.immutable_planes != 0);
    check_laws(report);
  }
}

TEMPLATE_TEST_CASE("plane arrangement laws: the products of a scene",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename laws_real_of<Int>::type;

  const auto scene = build_scene<Int, Real>();
  const auto &arm = GENERATE_REF(scene.stock, scene.refined);

  REQUIRE(arm.failed == 0);
  REQUIRE(arm.triangles != 0);
  // the premise the cut-incidence law needs: this scene states cuts
  REQUIRE(arm.report.cut_pieces != 0);
  REQUIRE(arm.report.immutable_planes + arm.report.local_planes ==
          arm.report.planes);
  check_laws(arm.report);
}

TEMPLATE_TEST_CASE("plane arrangement laws: the instrument can fail",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename laws_real_of<Int>::type;

  const auto scene = build_scene<Int, Real>();
  const auto &arm = GENERATE_REF(scene.stock, scene.refined);

  // dropping one emitted cut slot breaks the incidence law
  CHECK(arm.dropped.cut_plane_defects != 0);
  // hiding one published piece breaks the live-set law
  CHECK(arm.hidden.emitted_only != 0);
}

TEMPLATE_TEST_CASE("plane arrangement: two builds of one scene agree",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename laws_real_of<Int>::type;

  const auto first = build_scene<Int, Real>();
  const auto second = build_scene<Int, Real>();
  REQUIRE(first.stock.records == second.stock.records);
  REQUIRE(first.refined.records == second.refined.records);
  REQUIRE(first.refined.created == second.refined.created);
}

TEMPLATE_TEST_CASE("plane arrangement: a null refinement is the stock product",
                   "[cut][planes][laws]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename laws_real_of<Int>::type;

  const auto scene = build_scene<Int, Real>();
  REQUIRE(scene.stock.records.size() != 0);
  REQUIRE(scene.stock.records == scene.null_refined_records);
}
