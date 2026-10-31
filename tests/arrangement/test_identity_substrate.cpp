/**
 * @file test_identity_substrate.cpp
 * @brief The identity substrate's own statements over hand-built tables.
 *
 * One renumbering between two plane values (`extent_shift`, the extents
 * `make_diff_extents` reads), the local layer's moves composed into the
 * next value's evidence (`compose_local_moves`), the plane layer carried
 * into the next value's currency (`carry_plane_identities`), the
 * composed answer `P∘L` (`identity_substrate::composed_target`), the one
 * plane space the two tiers name (`identity_substrate::plane_name`) and the
 * account of every created identity (`account_created_identities`). Every
 * table is stated by hand, so each product is pinned row for row.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/identity/account_created_identities.hpp>
#include <trueform/arrangement/identity/carry_plane_identities.hpp>
#include <trueform/arrangement/identity/compose_local_moves.hpp>
#include <trueform/arrangement/identity/extent_shift.hpp>
#include <trueform/arrangement/identity/identity_substrate.hpp>
#include <trueform/arrangement/identity/plane_identity_tables.hpp>
#include <trueform/arrangement/planes/plane_flat_weld.hpp>
#include <trueform/arrangement/planes/plane_recovery_name.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/canonical_plane.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/meta.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace {

using substrate_index_t = int;
using substrate_row_t = std::array<substrate_index_t, 3>;
using substrate_weld_t = tf::arrangement::plane_flat_weld<substrate_index_t>;
using substrate_tables_t =
    tf::arrangement::plane_identity_tables<substrate_index_t, tf::exact::int32>;

template <typename T>
auto substrate_filled(std::initializer_list<T> values) -> tf::buffer<T> {
  tf::buffer<T> buffer;
  buffer.reserve(values.size());
  for (const auto &value : values)
    buffer.push_back(value);
  return buffer;
}

template <typename T>
auto substrate_rows(const tf::buffer<T> &buffer) -> std::vector<T> {
  return std::vector<T>(buffer.begin(), buffer.end());
}

auto substrate_weld_rows(const tf::buffer<substrate_weld_t> &welds)
    -> std::vector<substrate_row_t> {
  std::vector<substrate_row_t> rows;
  for (const auto &weld : welds)
    rows.push_back({weld.flat_0, weld.flat_1, weld.plane});
  return rows;
}

/// A prior value whose created space is `[10 | 3 promoted]` against a world
/// that now holds 14: the world appended `{10, 11, 12, 13}`, and the
/// promoted suffix `{10, 11, 12}` slides to `{14, 15, 16}`.
constexpr tf::arrangement::extent_shift<substrate_index_t> substrate_shift{
    10, 14, 13};

struct substrate_prior_value {
  auto n_base_created() const -> substrate_index_t { return 10; }
  auto n_base_planes() const -> substrate_index_t { return 3; }
  auto n_base_faces() const -> substrate_index_t { return 7; }
  auto promoted_frames() const -> std::vector<int> { return {0, 0}; }
  auto promoted_descriptors() const -> std::vector<int> { return {0}; }
};

struct substrate_world {
  using index_type = substrate_index_t;
  auto n_created_points() const -> substrate_index_t { return 14; }
  auto n_planes() const -> substrate_index_t { return 3; }
  auto n_faces() const -> substrate_index_t { return 9; }
};

/// The two-generation scenario `compose_local_moves` composes: 20 originals,
/// the local tier's created space `[8 | 2 appended]` and the prior plane
/// tier's four `{8, 9, 10, 11}` sliding to `{10, 11, 12, 13}`; three planes
/// and two promoted ones, the diff founding plane 3.
constexpr substrate_index_t substrate_n_flat = 20;
constexpr tf::arrangement::diff_extents<substrate_index_t> substrate_extents{
    {8, 10, 12}, {3, 4, 5}, {6, 6, 6}};

/// The closed local layer `L` the diff left standing: original 2 minted
/// into the carve's 9, original 5 and created 3 retired into created 1.
auto substrate_local_rows() -> tf::buffer<substrate_row_t> {
  return substrate_filled<substrate_row_t>({{0, 2, 9}, {0, 5, 1}, {1, 3, 1}});
}

/// The prior plane tier's folds, in the prior's flat currency: an original
/// onto a plane created identity (rides the carve), a local created identity
/// onto one, a plane created identity onto another, and a local created
/// identity `L` does not retire.
auto substrate_plane_welds() -> tf::buffer<substrate_weld_t> {
  return substrate_filled<substrate_weld_t>(
      {{2, 29, 1}, {23, 30, 4}, {28, 31, -1}, {24, 29, 2}});
}

auto substrate_compose(bool carved, tf::buffer<substrate_weld_t> &welds,
                       tf::buffer<substrate_index_t> &dirty_planes) -> void {
  const auto local_rows = substrate_local_rows();
  auto carve_moved = substrate_filled<substrate_row_t>({{0, 5, 1}, {0, 2, 9}});
  const auto census_moved =
      substrate_filled<substrate_row_t>({{1, 3, 1}, {1, 4, 6}});
  const auto plane_welds = substrate_plane_welds();
  tf::arrangement::compose_local_moves(
      local_rows, substrate_extents, substrate_n_flat, std::move(carve_moved),
      census_moved, plane_welds, carved, welds, dirty_planes);
}

auto substrate_name(std::array<substrate_index_t, 5> feature)
    -> substrate_tables_t::name_t {
  return tf::arrangement::make_plane_recovery_name<substrate_index_t>(
      feature, substrate_tables_t::param_t(0));
}

using substrate_plane_name_t = tf::exact::canonical_plane<tf::exact::int32>;

auto substrate_plane_name(int x, int offset) -> substrate_plane_name_t {
  using T2 = tf::exact::meta<tf::exact::int32>::T2;
  return {T2(x), T2(1), T2(0), T2(offset)};
}

auto substrate_prior_plane_tables() -> substrate_tables_t {
  substrate_tables_t prior;
  prior.created_class = substrate_filled<substrate_index_t>({0, 1, -1, 2});
  prior.direct_points =
      substrate_filled<substrate_tables_t::pt3_t>({{1, 2, 3}});
  prior.class_t = substrate_filled<substrate_tables_t::param_t>({3, 5, 7});
  prior.class_name = substrate_filled<substrate_tables_t::name_t>(
      {substrate_name({5, 25, 31, 3, 30}), substrate_name({6, 29, 21, 0, 0}),
       substrate_name({2, 30, 31, 0, 0})});
  prior.merges =
      substrate_filled<substrate_row_t>({{0, 9, 10}, {1, 9, 8}, {1, 11, 8}});
  prior.plane_names = substrate_filled<substrate_plane_name_t>(
      {substrate_plane_name(3, 7), substrate_plane_name(4, 9)});
  return prior;
}


using substrate_account_t = tf::arrangement::created_identity_account;
using substrate_triangle_t = std::array<substrate_index_t, 3>;

/// Four originals, the world's created `{0, 1, 2, 3}` and the plane tier's
/// `{4, 5}`. `L` retires 1 into the live 0 and 2 into 3, which nothing
/// references; `P` retires 5 into the live 4. The triangles name 0 and 4
/// past the originals, and so do the rows. Every table is stated through the
/// substrate's own verbs.
auto substrate_accounted()
    -> tf::arrangement::identity_substrate<substrate_index_t,
                                           tf::exact::int32> {
  tf::arrangement::identity_substrate<substrate_index_t, tf::exact::int32>
      identities;
  tf::buffer<substrate_row_t> moved;
  identities.weld_local(
      substrate_filled<substrate_row_t>({{1, 1, 0}, {1, 2, 3}}), 4, 4, moved);
  identities.weld_plane(substrate_filled<substrate_row_t>({{1, 5, 4}}));
  return identities;
}

auto substrate_accounted_triangles() -> tf::buffer<substrate_triangle_t> {
  return substrate_filled<substrate_triangle_t>({{0, 4, 8}, {1, 8, 2}});
}

auto substrate_accounts(
    const tf::arrangement::identity_substrate<substrate_index_t,
                                              tf::exact::int32> &identities,
    const tf::buffer<substrate_triangle_t> &triangles,
    std::initializer_list<substrate_index_t> named)
    -> std::vector<substrate_account_t> {
  const auto rows = substrate_filled<substrate_index_t>(named);
  return substrate_rows(tf::arrangement::account_created_identities(
      identities, tf::make_range(triangles), rows, substrate_index_t(4),
      substrate_index_t(6)));
}

} // namespace

TEST_CASE("extent_shift renumbers the suffix and holds the prefix",
          "[arrangement][identity]") {
  const auto &shift = substrate_shift;
  for (substrate_index_t id = 0; id < 10; ++id)
    CHECK(shift.forward(id) == id);
  CHECK(shift.forward(10) == 14);
  CHECK(shift.forward(11) == 15);
  CHECK(shift.forward(12) == 16);

  for (substrate_index_t id = 0; id < 10; ++id)
    CHECK(shift.inverse(id) == id);
  for (substrate_index_t id = 10; id < 14; ++id)
    CHECK(shift.inverse(id) == -1);
  CHECK(shift.inverse(14) == 10);
  CHECK(shift.inverse(16) == 12);

  std::vector<substrate_index_t> gap;
  for (const auto id : shift.gap())
    gap.push_back(id);
  CHECK(gap == std::vector<substrate_index_t>{10, 11, 12, 13});

  for (substrate_index_t id = 0; id < 13; ++id) {
    CHECK(shift.inverse(shift.forward(id)) == id);
    if (id > 0)
      CHECK(shift.forward(id - 1) < shift.forward(id));
  }
}

TEST_CASE("extent_shift lifted speaks flat currency past the originals",
          "[arrangement][identity]") {
  const auto lifted = substrate_shift.lifted(100);
  CHECK(lifted.prior == 110);
  CHECK(lifted.base == 114);
  CHECK(lifted.prior_end == 113);
  CHECK(lifted.forward(42) == 42);
  CHECK(lifted.forward(109) == 109);
  CHECK(lifted.forward(110) == 114);
  CHECK(lifted.forward(112) == 116);
  CHECK(lifted.inverse(111) == -1);
  CHECK(lifted.inverse(116) == 112);
}

TEST_CASE("extent_shift with nothing appended is the identity",
          "[arrangement][identity]") {
  const tf::arrangement::extent_shift<substrate_index_t> shift{6, 6, 9};
  for (substrate_index_t id = 0; id < 9; ++id) {
    CHECK(shift.forward(id) == id);
    CHECK(shift.inverse(id) == id);
  }
  CHECK(shift.gap().size() == 0);
}

TEST_CASE("make_diff_extents reads the prior's frozen extents",
          "[arrangement][identity]") {
  const auto extents = tf::arrangement::make_diff_extents(
      substrate_prior_value{}, substrate_index_t(3), substrate_world{});
  CHECK(extents.created.prior == 10);
  CHECK(extents.created.base == 14);
  CHECK(extents.created.prior_end == 13);
  CHECK(extents.planes.prior == 3);
  CHECK(extents.planes.base == 3);
  CHECK(extents.planes.prior_end == 5);
  CHECK(extents.faces.prior == 7);
  CHECK(extents.faces.base == 9);
  CHECK(extents.faces.prior_end == 8);
}

TEST_CASE("compose_local_moves states mirror rows then plane folds",
          "[arrangement][identity]") {
  auto welds = substrate_filled<substrate_weld_t>({{0, 1, 0}});
  auto dirty_planes = substrate_filled<substrate_index_t>({0});
  substrate_compose(true, welds, dirty_planes);

  CHECK(substrate_weld_rows(welds) ==
        std::vector<substrate_row_t>{// the caller's own, kept
                                     {0, 1, 0},
                                     // L's moves in row order, no witness
                                     {2, 29, -1},
                                     {5, 21, -1},
                                     {23, 21, -1},
                                     // the plane folds, in their order
                                     {29, 31, 1},
                                     {21, 32, 5},
                                     {30, 33, -1},
                                     {24, 31, 2}});
  CHECK(substrate_rows(dirty_planes) ==
        std::vector<substrate_index_t>{0, 3, 1, 5, 2});
}

TEST_CASE("compose_local_moves states no fold onto an uncarved original",
          "[arrangement][identity]") {
  tf::buffer<substrate_weld_t> welds;
  tf::buffer<substrate_index_t> dirty_planes;
  substrate_compose(false, welds, dirty_planes);

  CHECK(substrate_weld_rows(welds) ==
        std::vector<substrate_row_t>{{2, 29, -1},
                                     {5, 21, -1},
                                     {23, 21, -1},
                                     {21, 32, 5},
                                     {30, 33, -1},
                                     {24, 31, 2}});
  CHECK(substrate_rows(dirty_planes) ==
        std::vector<substrate_index_t>{3, 5, 2});
}

TEST_CASE("carry_plane_identities moves the plane layer into the next value",
          "[arrangement][identity]") {
  // the local tier's created space `[8 | 2 appended]`, the prior plane
  // tier's `{8, 9, 10, 11}` sliding to `{10, 11, 12, 13}`
  const tf::arrangement::extent_shift<substrate_index_t> created{8, 10, 12};
  substrate_tables_t identities;
  tf::arrangement::carry_plane_identities(substrate_prior_plane_tables(),
                                          created, substrate_n_flat, 5, 6,
                                          identities);

  CHECK(substrate_rows(identities.created_class) ==
        std::vector<substrate_index_t>{0, 1, -1, 2});
  REQUIRE(identities.direct_points.size() == 1);
  CHECK(identities.direct_points[0][2] == 3);
  CHECK(substrate_rows(identities.class_t) ==
        std::vector<substrate_tables_t::param_t>{3, 5, 7});
  // a crossing and a refinement name speak flat endpoints and cross the
  // lifted shift; any other kind is left as stated
  REQUIRE(identities.class_name.size() == 3);
  CHECK(identities.class_name[0] == substrate_name({5, 25, 33, 3, 32}));
  CHECK(identities.class_name[1] == substrate_name({6, 31, 21, 0, 0}));
  CHECK(identities.class_name[2] == substrate_name({2, 30, 31, 0, 0}));
  // an original's key never shifts, even one past the created extent; a
  // created key and every target do
  CHECK(substrate_rows(identities.merges) ==
        std::vector<substrate_row_t>{{0, 9, 12}, {1, 11, 10}, {1, 13, 10}});
  // a founded plane's name is geometry: it crosses verbatim, in place
  REQUIRE(identities.plane_names.size() == 2);
  CHECK((identities.plane_names[0] == substrate_plane_name(3, 7)));
  CHECK((identities.plane_names[1] == substrate_plane_name(4, 9)));
}

TEST_CASE("carry_plane_identities with nothing appended keeps every name",
          "[arrangement][identity]") {
  const tf::arrangement::extent_shift<substrate_index_t> created{8, 8, 12};
  substrate_tables_t identities;
  tf::arrangement::carry_plane_identities(substrate_prior_plane_tables(),
                                          created, substrate_n_flat, 5, 6,
                                          identities);
  REQUIRE(identities.class_name.size() == 3);
  CHECK(identities.class_name[0] == substrate_name({5, 25, 31, 3, 30}));
  CHECK(identities.class_name[1] == substrate_name({6, 29, 21, 0, 0}));
  CHECK(substrate_rows(identities.merges) ==
        std::vector<substrate_row_t>{{0, 9, 10}, {1, 9, 8}, {1, 11, 8}});
}

TEST_CASE("composed_target reads L first and P on what it left",
          "[arrangement][identity]") {
  tf::arrangement::identity_substrate<substrate_index_t, tf::exact::int32>
      identities;
  tf::buffer<substrate_row_t> moved;
  identities.weld_local(
      substrate_filled<substrate_row_t>({{0, 2, 9}, {0, 5, 1}, {1, 3, 1}}),
      substrate_n_flat, 10, moved);
  identities.weld_plane(
      substrate_filled<substrate_row_t>({{0, 6, 14}, {1, 1, 12}, {1, 20, 21}}));

  // a chain crossing both layers
  CHECK(identities.composed_target(0, 5) == 12);
  CHECK(identities.composed_target(1, 3) == 12);
  // L alone
  CHECK(identities.composed_target(0, 2) == 9);
  // P alone, on an original and on a created key
  CHECK(identities.composed_target(0, 6) == 14);
  CHECK(identities.composed_target(1, 20) == 21);
  CHECK(identities.composed_target(1, 1) == 12);
  // the source is part of the key: original 1 is not created 1
  CHECK(identities.composed_target(0, 1) == -1);
  // neither layer retires these
  CHECK(identities.composed_target(0, 7) == -1);
  CHECK(identities.composed_target(1, 30) == -1);
}

TEST_CASE("plane_name reads the local tier's planes, then the promotion's",
          "[arrangement][identity]") {
  tf::arrangement::identity_substrate<substrate_index_t, tf::exact::int32>
      identities;
  identities.seat_local_planes(substrate_filled<substrate_plane_name_t>(
      {substrate_plane_name(0, 1), substrate_plane_name(1, 2),
       substrate_plane_name(2, 3)}));
  identities.seat_founded_planes(substrate_filled<substrate_plane_name_t>(
      {substrate_plane_name(5, 8), substrate_plane_name(6, 9)}));

  CHECK((identities.plane_name(0) == substrate_plane_name(0, 1)));
  CHECK((identities.plane_name(2) == substrate_plane_name(2, 3)));
  // the promotion's planes stand past the local tier's extent
  CHECK((identities.plane_name(3) == substrate_plane_name(5, 8)));
  CHECK((identities.plane_name(4) == substrate_plane_name(6, 9)));

  const auto names = identities.plane_names();
  REQUIRE(names.size() == 5);
  for (substrate_index_t plane = 0; plane < 5; ++plane)
    CHECK((names[std::size_t(plane)] == identities.plane_name(plane)));

  // the local tier founding a plane slides the promotion's past it
  identities.seat_local_planes(
      substrate_filled<substrate_plane_name_t>({substrate_plane_name(7, 4)}));
  CHECK((identities.plane_name(3) == substrate_plane_name(7, 4)));
  CHECK((identities.plane_name(4) == substrate_plane_name(5, 8)));
  CHECK((identities.plane_name(5) == substrate_plane_name(6, 9)));
}

TEST_CASE("account_created_identities: live, retired or vanished",
          "[arrangement][identity]") {
  const auto accounts = substrate_accounts(
      substrate_accounted(), substrate_accounted_triangles(), {0, 4});
  CHECK(accounts ==
        std::vector<substrate_account_t>{
            substrate_account_t::live, substrate_account_t::retired,
            // its survivor is dead: a retirement is accounted
            // by the retirement, whatever it speaks
            substrate_account_t::retired, substrate_account_t::vanished,
            substrate_account_t::live, substrate_account_t::retired});
}

TEST_CASE("account_created_identities names each broken account",
          "[arrangement][identity]") {
  // a triangle naming the retired 1 is a dead corner; the 3 it also names
  // is referenced, so it lives
  auto emitting = substrate_accounted_triangles();
  emitting.push_back({0, 7, 5});
  const auto emitted =
      substrate_accounts(substrate_accounted(), emitting, {0, 4});
  CHECK(emitted[1] == substrate_account_t::emitted_dead);
  CHECK(emitted[2] == substrate_account_t::retired);
  CHECK(emitted[3] == substrate_account_t::live);

  // a row naming 3 that no triangle emits
  const auto named = substrate_accounts(
      substrate_accounted(), substrate_accounted_triangles(), {0, 3, 4});
  CHECK(named[2] == substrate_account_t::retired);
  CHECK(named[3] == substrate_account_t::unaccounted);
  CHECK(named[0] == substrate_account_t::live);
  CHECK(named[5] == substrate_account_t::retired);
}

TEST_CASE("account_created_identities derives the deaths",
          "[arrangement][identity]") {
  // four originals, the plane tier's `{0..4}`: 0 emitted; 1 unreferenced,
  // so dead by definition; 2 retired into the dead 1; 3 retired into the
  // live 0; 4 named by a row and emitted by no triangle
  tf::arrangement::identity_substrate<substrate_index_t, tf::exact::int32>
      identities;
  identities.weld_plane(
      substrate_filled<substrate_row_t>({{1, 2, 1}, {1, 3, 0}}));
  // corners are flat: created 0 is `n_flat + 0`
  const auto triangles = substrate_filled<substrate_triangle_t>({{4, 1, 2}});
  const auto rows = substrate_filled<substrate_index_t>({0, 4});
  const auto accounts =
      substrate_rows(tf::arrangement::account_created_identities(
          identities, tf::make_range(triangles), rows, substrate_index_t(4),
          substrate_index_t(5)));
  CHECK(accounts ==
        std::vector<substrate_account_t>{
            substrate_account_t::live, substrate_account_t::vanished,
            substrate_account_t::retired, substrate_account_t::retired,
            substrate_account_t::unaccounted});
}
