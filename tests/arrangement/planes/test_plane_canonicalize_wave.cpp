/**
 * @file test_plane_canonicalize_wave.cpp
 * @brief Tests for tf::arrangement::canonicalize_plane_wave and the wave it
 * closes
 *
 * The wave is three operations on one state — propose, canonicalize, commit —
 * run on the two-tier states the committed port produces. The cases here pin
 * what the canonicalize owns: equal children fusing into one key, a child
 * adopting a standing immutable group, a child superseding a standing local
 * one, the sweep a fresh-cut wave never runs, the no-op an empty wave costs,
 * and the in-place substitution a non-diff carrier takes.
 *
 * On the states the committed weld surgery leaves, the same operations close
 * the moved keys: two meeting each other, one meeting a split child, one
 * meeting an untouched world group, one meeting nothing and keeping its
 * identity, and the split-only wave that is byte for byte the wave it always
 * was.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/reallocate.hpp>
#include <trueform/arrangement/planes/canonicalize_plane_wave.hpp>
#include <trueform/arrangement/planes/commit_plane_wave_blocks.hpp>
#include <trueform/arrangement/planes/finalize_plane_piece_tickets.hpp>
#include <trueform/arrangement/planes/make_plane_final_piece_definitions.hpp>
#include <trueform/arrangement/planes/plane_diff.hpp>
#include <trueform/arrangement/planes/plane_piece_key.hpp>
#include <trueform/arrangement/planes/port_plane_diff.hpp>
#include <trueform/arrangement/planes/propose_plane_split_pieces.hpp>
#include <trueform/arrangement/planes/plane_tier_definitions.hpp>
#include <trueform/arrangement/planes/weld_local_plane_rows.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>
#include <trueform/intersect/graph/plane_identity_collapse.hpp>
#include <trueform/intersect/graph/plane_tables.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace {

/// The index type each lattice width carries its identities in.
template <typename Int> struct index_of;
template <> struct index_of<tf::exact::int32> {
  using type = int;
};
template <> struct index_of<tf::exact::int64> {
  using type = long long;
};

template <typename Index>
using canonicalize_def_t = tf::intersect::graph::plane_edge_def<Index>;

/// `mark` fills every field a piece must inherit verbatim; `flags` is stated
/// per instance because the piece maker rewrites exactly those bits.
template <typename Index>
auto make_def(std::int16_t point_tag_0, Index point_0, std::int16_t point_tag_1,
              Index point_1, Index id, Index face, int mark, int flags)
    -> canonicalize_def_t<Index> {
  canonicalize_def_t<Index> def{};
  def.point_0 = point_0;
  def.point_1 = point_1;
  def.id = id;
  def.face = face;
  def.object_other = Index(100 + mark);
  def.point_tag_0 = point_tag_0;
  def.point_tag_1 = point_tag_1;
  def.tag_other = std::int16_t(mark);
  def.ordinal = std::int16_t(10 + mark);
  def.side = std::int16_t(mark % 3);
  def.flags = std::uint8_t(flags);
  return def;
}

template <typename Index, typename Int>
auto fill_tables(tf::intersect::graph::plane_tables<Index, Int> &tables,
                 std::initializer_list<canonicalize_def_t<Index>> defs,
                 std::initializer_list<int> def_offsets,
                 std::initializer_list<int> block_offsets,
                 std::initializer_list<int> block_rows) -> void {
  for (const auto &def : defs)
    tables.defs().push_back(def);
  for (const auto offset : def_offsets)
    tables.def_offsets().push_back(Index(offset));
  for (const auto offset : block_offsets)
    tables.edges().offsets_buffer().push_back(Index(offset));
  for (const auto row : block_rows)
    tables.edges().data_buffer().push_back(Index(row));
  tables.n_canon() = Index(def_offsets.size()) - Index(1);
}

template <typename Index>
auto fill_buffer(tf::buffer<Index> &buffer, std::initializer_list<int> values)
    -> void {
  buffer.clear();
  for (const auto value : values)
    buffer.push_back(Index(value));
}

template <typename Index, typename Values>
auto values_are(const Values &values, std::initializer_list<int> expected)
    -> bool {
  if (values.size() != expected.size())
    return false;
  std::size_t i = 0;
  for (const auto value : expected)
    if (values[i++] != Index(value))
      return false;
  return true;
}

template <typename Index>
auto defs_equal(const canonicalize_def_t<Index> &x,
                const canonicalize_def_t<Index> &y) -> bool {
  return x.point_0 == y.point_0 && x.point_1 == y.point_1 && x.id == y.id &&
         x.face == y.face && x.object_other == y.object_other &&
         x.point_tag_0 == y.point_tag_0 && x.point_tag_1 == y.point_tag_1 &&
         x.tag_other == y.tag_other && x.ordinal == y.ordinal &&
         x.side == y.side && x.flags == y.flags;
}

template <typename Index>
auto same_values(const tf::buffer<Index> &x, const tf::buffer<Index> &y)
    -> bool {
  if (x.size() != y.size())
    return false;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (x[i] != y[i])
      return false;
  return true;
}

/// THE WORLD: four faces on four planes of their own, three canonical groups
/// in endpoint-key order — `AX (-1,0)-(0,0)` carried TWICE (faces 2 and 3),
/// `AB (0,0)-(0,1)` once (face 0), `AD (0,0)-(0,2)` once (face 1). AX names
/// the created identity `0`, which is what makes it collidable: a split of AB
/// or AD at created `0` states a child with AX's key.
template <typename Index, typename Int>
auto make_world() -> tf::intersect::graph::plane_tables<Index, Int> {
  tf::intersect::graph::plane_tables<Index, Int> world;
  fill_tables<Index, Int>(world,
                          {make_def<Index>(-1, 0, 0, 0, 0, 2, 2, 0),
                           make_def<Index>(-1, 0, 0, 0, 0, 3, 3, 0),
                           make_def<Index>(0, 0, 0, 1, 1, 0, 0, 0),
                           make_def<Index>(0, 0, 0, 2, 2, 1, 1, 0)},
                          {0, 2, 3, 4}, {0, 1, 2, 3, 4}, {2, 3, 0, 1});
  return world;
}

/// THE TWO-LOSER WORLD: two parents on their own planes, each cut at a
/// PRE-EXISTING created identity a standing LOCAL group already names, and both
/// of those standing groups carried by plane 2 — a plane no split parent
/// touches, so it takes the in-place substitution path for both of them at
/// once. The inert groups are that carrier's other rows, which the
/// substitution must leave exactly where they are.
template <typename Index, typename Int>
auto make_two_loser_world(std::size_t inert_groups)
    -> tf::intersect::graph::plane_tables<Index, Int> {
  tf::intersect::graph::plane_tables<Index, Int> world;
  world.def_offsets().push_back(Index(0));
  world.defs().push_back(make_def<Index>(-1, 0, 0, 0, 0, 2, 0, 0));
  world.def_offsets().push_back(Index(world.defs().size()));
  world.defs().push_back(make_def<Index>(-1, 1, 0, 2, 1, 2, 1, 0));
  world.def_offsets().push_back(Index(world.defs().size()));
  world.defs().push_back(make_def<Index>(0, 0, 0, 3, 2, 0, 2, 0));
  world.def_offsets().push_back(Index(world.defs().size()));
  world.defs().push_back(make_def<Index>(0, 2, 0, 4, 3, 1, 3, 0));
  world.def_offsets().push_back(Index(world.defs().size()));
  for (std::size_t at = 0; at < inert_groups; ++at) {
    const auto point = Index(10 + 2 * at);
    world.defs().push_back(make_def<Index>(0, point, 0, point + Index(1),
                                           Index(4 + at), 2,
                                           int(at % std::size_t(30000)), 0));
    world.def_offsets().push_back(Index(world.defs().size()));
  }
  world.n_canon() = Index(4 + inert_groups);
  world.edges().offsets_buffer().push_back(Index(0));
  world.edges().data_buffer().push_back(Index(2));
  world.edges().offsets_buffer().push_back(Index(1));
  world.edges().data_buffer().push_back(Index(3));
  world.edges().offsets_buffer().push_back(Index(2));
  world.edges().data_buffer().push_back(Index(0));
  world.edges().data_buffer().push_back(Index(1));
  for (std::size_t at = 0; at < inert_groups; ++at)
    world.edges().data_buffer().push_back(Index(4 + at));
  world.edges().offsets_buffer().push_back(
      Index(world.edges().data_buffer().size()));
  return world;
}

/// THE WELD WORLD: five faces on five planes of their own, five canonical
/// groups in endpoint-key order — `(-1,0)-(0,1)` on face 3, `(0,0)-(0,1)` on
/// face 0, `(0,0)-(0,3)` on face 4, `(0,1)-(0,2)` on face 1 and
/// `(0,1)-(0,3)` on face 2. A merge row moves a key here by absorbing an
/// original vertex into a created identity, and WHICH identity absorbs WHICH
/// vertex decides what the moved key then meets: another moved key, a split
/// child, an untouched world group, or nothing at all.
template <typename Index, typename Int>
auto make_weld_world() -> tf::intersect::graph::plane_tables<Index, Int> {
  tf::intersect::graph::plane_tables<Index, Int> world;
  fill_tables<Index, Int>(world,
                          {make_def<Index>(-1, 0, 0, 1, 0, 3, 0, 0),
                           make_def<Index>(0, 0, 0, 1, 1, 0, 1, 2),
                           make_def<Index>(0, 0, 0, 3, 2, 4, 2, 2),
                           make_def<Index>(0, 1, 0, 2, 3, 1, 3, 1),
                           make_def<Index>(0, 1, 0, 3, 4, 2, 4, 4)},
                          {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5},
                          {1, 3, 4, 0, 2});
  return world;
}

/// THE SHARED BYSTANDER: one group carried by TWO planes, and a second group
/// only the first of them carries. A wave that changes the second must port
/// its carrier and must NOT touch the shared one — which is the whole of the
/// ownership law's second clause.
template <typename Index, typename Int>
auto make_shared_bystander_world()
    -> tf::intersect::graph::plane_tables<Index, Int> {
  tf::intersect::graph::plane_tables<Index, Int> world;
  fill_tables<Index, Int>(world,
                          {make_def<Index>(-1, 0, 0, 0, 0, 0, 0, 0),
                           make_def<Index>(-1, 0, 0, 0, 0, 1, 1, 0),
                           make_def<Index>(0, 1, 0, 2, 1, 0, 2, 0)},
                          {0, 2, 3}, {0, 2, 3}, {0, 2, 1});
  return world;
}

template <typename Index> auto plane_of_face(Index face) -> Index {
  return face;
}

template <typename Index, typename Int> struct state_t {
  tf::intersect::graph::plane_tables<Index, Int> local;
  tf::buffer<Index> plane_ticket;
  tf::buffer<Index> group_router;
};

template <typename Index> struct wave_t {
  tf::buffer<Index> parents;
  tf::buffer<Index> offsets;
  tf::buffer<Index> data;
  Index created_mint_base = 0;
};

template <typename Index>
auto make_wave(std::initializer_list<int> parents,
               std::initializer_list<int> offsets,
               std::initializer_list<int> data, int created_mint_base)
    -> wave_t<Index> {
  wave_t<Index> wave;
  fill_buffer(wave.parents, parents);
  fill_buffer(wave.offsets, offsets);
  fill_buffer(wave.data, data);
  wave.created_mint_base = Index(created_mint_base);
  return wave;
}

/// Everything the three operations publish between themselves.
template <typename Index> struct products_t {
  tf::buffer<tf::arrangement::plane_split_piece_layout<Index>> layout;
  tf::buffer<canonicalize_def_t<Index>> proposals;
  tf::buffer<Index> carriers;
  tf::buffer<Index> probe;
  tf::buffer<Index> final_row;
  tf::buffer<std::array<Index, 2>> losers;
};

/// Port planes, taking the named groups the caller says CHANGE. Taking every
/// group a plane names is the state a wave that changed all of them leaves.
template <typename Index, typename Int>
auto port_taking(const tf::intersect::graph::plane_tables<Index, Int> &world,
                 state_t<Index, Int> &state, std::initializer_list<int> diff,
                 const tf::buffer<Index> &taken) -> bool {
  tf::buffer<Index> planes;
  fill_buffer(planes, diff);
  return tf::arrangement::port_plane_diff(world, tf::make_range(planes), taken,
                                          state.local, state.plane_ticket,
                                          state.group_router);
}

/// Port planes taking the named groups, with RIDERS: the definitions a face
/// the cut world never named states on a group it joins instead of founding.
/// `rider_offsets` is aligned with `taken`, exactly as the port requires.
template <typename Index, typename Int>
auto port_riding(const tf::intersect::graph::plane_tables<Index, Int> &world,
                 state_t<Index, Int> &state, std::initializer_list<int> diff,
                 const tf::buffer<Index> &taken,
                 const tf::buffer<canonicalize_def_t<Index>> &riders,
                 std::initializer_list<int> rider_offsets) -> bool {
  tf::buffer<Index> planes;
  fill_buffer(planes, diff);
  tf::buffer<Index> offsets;
  fill_buffer(offsets, rider_offsets);
  return tf::arrangement::port_plane_diff(world, tf::make_range(planes), taken,
                                          state.local, state.plane_ticket,
                                          state.group_router, offsets, riders);
}

template <typename Index, typename Int>
auto port(const tf::intersect::graph::plane_tables<Index, Int> &world,
          state_t<Index, Int> &state, std::initializer_list<int> diff) -> bool {
  tf::buffer<Index> taken;
  for (const auto plane : diff)
    for (const auto row : world.plane_edges(Index(plane))) {
      const auto group = world.edge_defs()[std::size_t(row)].id;
      if (state.group_router.size() == 0 ||
          state.group_router[std::size_t(group)] == Index(-1))
        taken.push_back(group);
    }
  std::sort(taken.begin(), taken.end());
  taken.erase_till_end(std::unique(taken.begin(), taken.end()));
  return port_taking(world, state, diff, taken);
}

template <typename Index>
auto state_probe(const wave_t<Index> &wave, tf::buffer<Index> &probe) -> void {
  probe.clear();
  for (const auto cut : wave.data)
    if (cut < wave.created_mint_base)
      probe.push_back(cut);
  std::sort(probe.begin(), probe.end());
  probe.erase_till_end(std::unique(probe.begin(), probe.end()));
}

/// THE WAVE: propose, canonicalize, commit — the three operations in the one
/// order the design states, on the one state they share.
template <typename Index, typename Int, typename PlaneOfFace>
auto run_wave(const tf::intersect::graph::plane_tables<Index, Int> &world,
              state_t<Index, Int> &state, const wave_t<Index> &wave,
              const PlaneOfFace &plane_of, products_t<Index> &out) -> bool {
  const tf::buffer<Index> no_changed_groups;
  state_probe(wave, out.probe);
  if (!tf::arrangement::propose_plane_split_pieces(
          wave.parents, wave.offsets, wave.data, plane_of, state.local,
          state.plane_ticket, out.layout, out.proposals, out.carriers))
    return false;
  if (!tf::arrangement::canonicalize_plane_wave(
          world, out.layout, out.proposals, out.probe, state.plane_ticket,
          plane_of, state.local, state.group_router, out.final_row, out.losers,
          no_changed_groups))
    return false;
  return tf::arrangement::commit_plane_wave_blocks(
      world, out.layout, out.final_row, out.carriers, out.losers, plane_of,
      state.local, state.plane_ticket);
}

/// Everything one weld states and everything the surgery answers, plus the
/// merge TARGETS: pre-existing identities by definition, which is what makes a
/// standing collision with a moved key discoverable by the same sweep.
template <typename Index> struct weld_t {
  tf::buffer<Index> retired;
  tf::buffer<std::array<Index, 3>> merges;
  tf::buffer<Index> targets;
  tf::buffer<Index> row_offsets;
  /// the piece TICKET each touched row names, in sweep order
  tf::buffer<Index> row_groups;
  tf::buffer<Index> frontier;
  tf::buffer<Index> recdt_planes;
  tf::buffer<Index> changed;
};

template <typename Index> auto weld_vertex_offsets() -> tf::buffer<Index> {
  tf::buffer<Index> offsets;
  offsets.push_back(Index(0));
  offsets.push_back(Index(4));
  return offsets;
}

/// The wave's rewrite rows, closed by their own producer: an original vertex
/// speaks the created identity that absorbed it.
template <typename Index>
auto state_merges(weld_t<Index> &weld,
                  std::initializer_list<std::array<int, 2>> rows) -> void {
  for (const auto &row : rows) {
    weld.retired.push_back(Index(row[0]));
    weld.merges.push_back({Index(0), Index(row[0]), Index(row[1])});
    weld.targets.push_back(Index(row[1]));
  }
  tf::intersect::graph::close_plane_merges(weld.merges);
  std::sort(weld.targets.begin(), weld.targets.end());
  weld.targets.erase_till_end(
      std::unique(weld.targets.begin(), weld.targets.end()));
}

/// The world these cases state: their tables and nothing else, so the diff
/// names no carriers and sweeps every one of them.
template <typename Index, typename Int> struct tables_world_t {
  const tf::intersect::graph::plane_tables<Index, Int> *_tables;

  auto tables() const
      -> const tf::intersect::graph::plane_tables<Index, Int> & {
    return *_tables;
  }
};

/// THE WELD WAVE, in the order the wire states it: the committed gather sweeps
/// the touched rows, the committed surgery rewrites them, and the same three
/// operations follow — the probe unioned by the caller, the surgery's
/// survivors entering the canonicalize beside the children.
template <typename Index, typename Int, typename PlaneOfFace>
auto run_welded_wave(
    const tf::intersect::graph::plane_tables<Index, Int> &world,
    state_t<Index, Int> &state, const wave_t<Index> &wave, weld_t<Index> &weld,
    const tf::buffer<Index> &probe_targets, const PlaneOfFace &plane_of,
    products_t<Index> &out) -> bool {
  const auto vertex_offsets = weld_vertex_offsets<Index>();
  const tf::buffer<Index> split_edge;
  const tf::buffer<Index> split_tier;
  tf::buffer<Index> taken;
  if (!tf::arrangement::gather_plane_diff(
          tables_world_t<Index, Int>{&world}, state.local, state.plane_ticket,
          state.group_router, weld.retired, vertex_offsets, split_edge,
          split_tier, plane_of, weld.row_offsets, weld.row_groups,
          weld.frontier, taken) ||
      !tf::arrangement::weld_local_plane_rows(
          world, weld.merges, vertex_offsets, weld.row_offsets, weld.row_groups,
          plane_of, state.local, state.plane_ticket, state.group_router,
          weld.recdt_planes, weld.changed))
    return false;
  state_probe(wave, out.probe);
  if (!tf::arrangement::propose_plane_split_pieces(
          wave.parents, wave.offsets, wave.data, plane_of, state.local,
          state.plane_ticket, out.layout, out.proposals, out.carriers))
    return false;
  tf::core::append(probe_targets, out.probe);
  std::sort(out.probe.begin(), out.probe.end());
  out.probe.erase_till_end(std::unique(out.probe.begin(), out.probe.end()));
  if (!tf::arrangement::canonicalize_plane_wave(
          world, out.layout, out.proposals, out.probe, state.plane_ticket,
          plane_of, state.local, state.group_router, out.final_row, out.losers,
          weld.changed))
    return false;
  return tf::arrangement::commit_plane_wave_blocks(
      world, out.layout, out.final_row, out.carriers, out.losers, plane_of,
      state.local, state.plane_ticket);
}

template <typename Index, typename Int> struct world_tier_t {
  const tf::intersect::graph::plane_tables<Index, Int> *tables;

  auto n_canon() const -> Index { return tables->n_canon(); }
  auto edge_defs() const { return tables->edge_defs(); }
  auto canon_group(Index group) const { return tables->canon_group(group); }
  auto n_planes() const -> Index { return Index(tables->edges().size()); }
  auto plane_edges(Index plane) const { return tables->plane_edges(plane); }
};

template <typename Index, typename Int> struct local_tier_t {
  const tf::intersect::graph::plane_tables<Index, Int> *tables;

  auto n_canon() const -> Index { return tables->n_canon(); }
  auto edge_defs() const { return tables->edge_defs(); }
  auto canon_group(Index group) const { return tables->canon_group(group); }
  auto n_planes() const -> Index { return Index(tables->edges().size()); }
  auto plane_edges(Index block) const { return tables->plane_edges(block); }
};

/// THE BOUNDARY: the piece definitions and the ticket resolution, run on the
/// state the wave produced. One triangle per plane, its first slot naming the
/// pair the case probes; a slot on an ACTIVE carrier starts at `-1` and is
/// answered by key, a slot on an untouched carrier starts at its immutable
/// group and is answered by the router.
template <typename Index, typename Int>
auto publish(
    const tf::intersect::graph::plane_tables<Index, Int> &world,
    const state_t<Index, Int> &state,
    const tf::buffer<std::array<Index, 3>> &triangles,
    tf::buffer<std::array<Index, 3>> &slots,
    tf::offset_block_buffer<Index, canonicalize_def_t<Index>> &definitions)
    -> bool {
  tf::buffer<Index> vertex_offsets;
  vertex_offsets.push_back(Index(0));
  vertex_offsets.push_back(Index(4));
  tf::buffer<Index> plane_offsets;
  for (Index plane = 0; plane <= Index(world.edges().size()); ++plane)
    plane_offsets.push_back(plane);
  const world_tier_t<Index, Int> immutable{&world};
  const local_tier_t<Index, Int> current{&state.local};
  return tf::arrangement::make_plane_final_piece_definitions(
             immutable, current, world.n_canon(), definitions) &&
         tf::arrangement::finalize_plane_piece_tickets(
             immutable, current,
             tf::arrangement::make_plane_tier_definitions(world, state.local,
                                                          true),
             state.group_router, state.plane_ticket, vertex_offsets, Index(4),
             world.n_canon(), triangles, slots, plane_offsets);
}

template <typename Index>
auto make_triples(std::initializer_list<std::array<int, 3>> values)
    -> tf::buffer<std::array<Index, 3>> {
  tf::buffer<std::array<Index, 3>> out;
  for (const auto &value : values)
    out.push_back({Index(value[0]), Index(value[1]), Index(value[2])});
  return out;
}

/// The sweep states a piece TICKET per touched row: a group still the world's
/// by its own id, one this arrangement owns past the immutable extent. The
/// fixtures name the LOCAL groups and the extent lifts them.
template <typename Index, typename Int>
auto local_tickets_are(
    const tf::intersect::graph::plane_tables<Index, Int> &world,
    const tf::buffer<Index> &values, std::initializer_list<int> local_groups)
    -> bool {
  if (values.size() != local_groups.size())
    return false;
  auto at = values.begin();
  for (const auto group : local_groups)
    if (*at++ != world.n_canon() + Index(group))
      return false;
  return true;
}

template <typename Index, typename Int>
auto block_is(const state_t<Index, Int> &state, int plane,
              std::initializer_list<int> expected) -> bool {
  if (std::size_t(plane) >= state.plane_ticket.size())
    return false;
  const auto block = state.plane_ticket[std::size_t(plane)];
  if (block < Index(0) || std::size_t(block) >= state.local.edges().size())
    return false;
  return values_are<Index>(state.local.plane_edges(block), expected);
}

/// A published span, instance by instance: the faces it unions, in the order
/// the boundary states them.
template <typename Index>
auto span_faces_are(
    const tf::offset_block_buffer<Index, canonicalize_def_t<Index>>
        &definitions,
    int group, std::initializer_list<int> faces) -> bool {
  if (std::size_t(group) >= definitions.size())
    return false;
  const auto span = tf::make_range(definitions)[std::size_t(group)];
  if (span.size() != faces.size())
    return false;
  std::size_t at = 0;
  for (const auto face : faces)
    if (span[at++].face != Index(face))
      return false;
  return true;
}

/// Every instance of a group states ONE key, and a group that kept its
/// identity states its own id on every one of them.
template <typename Index, typename Int>
auto group_key_is(const state_t<Index, Int> &state, int group,
                  std::initializer_list<int> key) -> bool {
  if (Index(group) < Index(0) || Index(group) >= state.local.n_canon() ||
      key.size() != 4)
    return false;
  const auto span = state.local.canon_group(Index(group));
  if (span.size() == 0)
    return false;
  const auto expected = key.begin();
  for (const auto &def : span)
    if (def.id != Index(group) ||
        Index(def.point_tag_0) != Index(expected[0]) ||
        def.point_0 != Index(expected[1]) ||
        Index(def.point_tag_1) != Index(expected[2]) ||
        def.point_1 != Index(expected[3]))
      return false;
  return true;
}

template <typename Index>
auto losers_are(const products_t<Index> &out,
                std::initializer_list<std::array<int, 2>> expected) -> bool {
  if (out.losers.size() != expected.size())
    return false;
  std::size_t at = 0;
  for (const auto &loser : expected) {
    if (out.losers[at][0] != Index(loser[0]) ||
        out.losers[at][1] != Index(loser[1]))
      return false;
    ++at;
  }
  return true;
}

/// Two runs of one wave, buffer by buffer: the state both operations left and
/// everything they published.
template <typename Index, typename Int>
auto runs_match(const state_t<Index, Int> &x, const products_t<Index> &x_out,
                const state_t<Index, Int> &y, const products_t<Index> &y_out)
    -> bool {
  if (x.local.n_canon() != y.local.n_canon() ||
      x.local.defs().size() != y.local.defs().size() ||
      x_out.losers.size() != y_out.losers.size())
    return false;
  for (std::size_t row = 0; row < x.local.defs().size(); ++row)
    if (!defs_equal(x.local.defs()[row], y.local.defs()[row]))
      return false;
  for (std::size_t at = 0; at < x_out.losers.size(); ++at)
    if (x_out.losers[at][0] != y_out.losers[at][0] ||
        x_out.losers[at][1] != y_out.losers[at][1])
      return false;
  return same_values(x.local.def_offsets(), y.local.def_offsets()) &&
         same_values(x.local.edges().offsets_buffer(),
                     y.local.edges().offsets_buffer()) &&
         same_values(x.local.edges().data_buffer(),
                     y.local.edges().data_buffer()) &&
         same_values(x.plane_ticket, y.plane_ticket) &&
         same_values(x.group_router, y.group_router) &&
         same_values(x_out.final_row, y_out.final_row) &&
         same_values(x_out.probe, y_out.probe);
}

template <typename Index, typename Int>
auto group_faces_are(const state_t<Index, Int> &state, int group,
                     std::initializer_list<int> faces) -> bool {
  if (Index(group) >= state.local.n_canon())
    return false;
  const auto span = state.local.canon_group(Index(group));
  if (span.size() != faces.size())
    return false;
  std::size_t at = 0;
  for (const auto face : faces)
    if (span[at++].face != Index(face))
      return false;
  return true;
}

/// A live block is KEY-ascending: the invariant every block consumer indexes
/// by. Equal keys are one group, so their rows need no order among them.
template <typename Index, typename Int>
auto blocks_are_key_ordered(const state_t<Index, Int> &state) -> bool {
  const auto defs = state.local.edge_defs();
  for (std::size_t plane = 0; plane < state.plane_ticket.size(); ++plane) {
    const auto block = state.plane_ticket[plane];
    if (block == Index(-1))
      continue;
    if (block < Index(0) || std::size_t(block) >= state.local.edges().size())
      return false;
    const auto rows = state.local.plane_edges(block);
    for (std::size_t at = 1; at < rows.size(); ++at) {
      const auto previous = tf::arrangement::plane_piece_key<Index>(
          defs[std::size_t(rows[at - 1])]);
      const auto next =
          tf::arrangement::plane_piece_key<Index>(defs[std::size_t(rows[at])]);
      if (previous > next)
        return false;
      if (previous == next &&
          defs[std::size_t(rows[at - 1])].id != defs[std::size_t(rows[at])].id)
        return false;
    }
  }
  return true;
}

template <typename Index, typename Int>
auto allocates_nothing(const state_t<Index, Int> &state,
                       const products_t<Index> &out) -> bool {
  return state.local.defs().capacity() == 0 &&
         state.local.def_offsets().capacity() == 0 &&
         state.local.edges().offsets_buffer().capacity() == 0 &&
         state.local.edges().data_buffer().capacity() == 0 &&
         state.plane_ticket.capacity() == 0 &&
         state.group_router.capacity() == 0 &&
         state.local.n_canon() == Index(0) && out.layout.size() == 0 &&
         out.proposals.size() == 0 && out.carriers.size() == 0 &&
         out.probe.size() == 0 && out.final_row.size() == 0 &&
         out.losers.size() == 0;
}

} // namespace

TEMPLATE_TEST_CASE("plane wave: equal children of two parents are one key",
                   "[cut][planes][canonicalize]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // `AB` and `AD` are independent parents on their own planes, both cut at the
  // same created identity: their two `AX'` children are ONE key, so they are
  // ONE group carrying BOTH provenances, and both carriers publish ONE ticket
  // for it.
  const auto world = make_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  REQUIRE(port(world, state, {0, 1}));
  REQUIRE(run_wave(
      world, state, make_wave<Index>({0, 1}, {0, 1, 2}, {5, 5}, 6),
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(state.local.defs().size() == 6);
  REQUIRE(state.local.n_canon() == Index(5));
  REQUIRE(values_are<Index>(state.local.def_offsets(), {0, 1, 2, 4, 5, 6}));
  REQUIRE(group_faces_are(state, 2, {0, 1}));
  REQUIRE(group_faces_are(state, 3, {0}));
  REQUIRE(group_faces_are(state, 4, {1}));
  REQUIRE(values_are<Index>(out.final_row, {2, 4, 3, 5}));
  REQUIRE(out.losers.size() == 0);
  REQUIRE(values_are<Index>(state.group_router, {-1, -2, -2}));
  REQUIRE(block_is(state, 0, {2, 4}));
  REQUIRE(block_is(state, 1, {3, 5}));
  REQUIRE(values_are<Index>(state.plane_ticket, {2, 3, -1, -1}));
  REQUIRE(blocks_are_key_ordered(state));

  const auto triangles =
      make_triples<Index>({{9, 0, 3}, {9, 0, 3}, {4, 0, 3}, {4, 0, 3}});
  auto slots = make_triples<Index>(
      {{-1, -1, -1}, {-1, -1, -1}, {0, -1, -1}, {0, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[0][0] == Index(5));
  REQUIRE(slots[1][0] == Index(5));
  REQUIRE(slots[2][0] == Index(0));
  REQUIRE(slots[3][0] == Index(0));
  REQUIRE(span_faces_are(definitions, 2, {0, 1}));
}

TEMPLATE_TEST_CASE("plane wave: a child equal to a standing group fuses",
                   "[cut][planes][canonicalize]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // `AB` is split at the very identity `AX` already names. THE OWNERSHIP LAW
  // promoted AX's carriers into the wave's port, so by the time keys close AX
  // is a standing LOCAL group: the winner mints from the union of the child
  // and AX's instances, the loser retires structurally, and the router
  // COMPOSES the root onto the winner.
  const auto world = make_world<Index, Int>();

  // driving the wave against the unpromoted state is a state the arrangement
  // cannot be in, and the canonicalize rejects it
  state_t<Index, Int> unpromoted;
  products_t<Index> rejected;
  REQUIRE(port(world, unpromoted, {0}));
  REQUIRE_FALSE(run_wave(
      world, unpromoted, make_wave<Index>({0}, {0, 1}, {0}, 1),
      [](Index face) { return plane_of_face(face); }, rejected));

  state_t<Index, Int> state;
  products_t<Index> out;
  REQUIRE(port(world, state, {0, 2, 3}));
  REQUIRE(run_wave(
      world, state, make_wave<Index>({1}, {0, 1}, {0}, 1),
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(out.probe.size() == 1);
  REQUIRE(state.local.defs().size() == 7);
  REQUIRE(state.local.n_canon() == Index(4));
  REQUIRE(values_are<Index>(state.local.def_offsets(), {0, 2, 3, 6, 7}));
  REQUIRE(group_faces_are(state, 2, {0, 2, 3}));
  REQUIRE(out.losers.size() == 1);
  REQUIRE(out.losers[0] == std::array<Index, 2>{Index(0), Index(2)});
  REQUIRE(values_are<Index>(out.final_row, {3, 6}));
  REQUIRE(values_are<Index>(state.group_router, {2, -2, -1}));
  REQUIRE(block_is(state, 0, {3, 6}));
  REQUIRE(block_is(state, 2, {4}));
  REQUIRE(block_is(state, 3, {5}));
  REQUIRE(values_are<Index>(state.plane_ticket, {3, -1, 1, 2}));
  REQUIRE(blocks_are_key_ordered(state));

  const auto triangles =
      make_triples<Index>({{4, 0, 3}, {5, 0, 3}, {4, 0, 3}, {4, 0, 3}});
  auto slots = make_triples<Index>(
      {{-1, -1, -1}, {-1, -1, -1}, {0, -1, -1}, {0, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[0][0] == Index(5));
  REQUIRE(slots[2][0] == Index(5));
  REQUIRE(slots[3][0] == Index(5));
  REQUIRE(span_faces_are(definitions, 2, {0, 2, 3}));
}

TEMPLATE_TEST_CASE("plane wave: a fresh-cut-only wave never runs the sweep",
                   "[cut][planes][canonicalize]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // Every cut is an identity this wave minted, so no standing row can name
  // one: the probe is EMPTY and the sweep never runs. The children still fuse
  // among themselves, and the state is the one the probed wave reaches — byte
  // for byte.
  const auto world = make_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  REQUIRE(port(world, state, {0, 1}));
  REQUIRE(run_wave(
      world, state, make_wave<Index>({0, 1}, {0, 1, 2}, {5, 5}, 5),
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(out.probe.size() == 0);
  REQUIRE(out.losers.size() == 0);
  REQUIRE(state.local.defs().size() == 6);
  REQUIRE(state.local.n_canon() == Index(5));
  REQUIRE(group_faces_are(state, 2, {0, 1}));
  REQUIRE(values_are<Index>(state.group_router, {-1, -2, -2}));
  REQUIRE(block_is(state, 0, {2, 4}));
  REQUIRE(block_is(state, 1, {3, 5}));

  state_t<Index, Int> probed;
  products_t<Index> probed_out;
  REQUIRE(port(world, probed, {0, 1}));
  REQUIRE(run_wave(
      world, probed, make_wave<Index>({0, 1}, {0, 1, 2}, {5, 5}, 6),
      [](Index face) { return plane_of_face(face); }, probed_out));
  REQUIRE(probed_out.probe.size() == 1);
  REQUIRE(same_values(state.local.def_offsets(), probed.local.def_offsets()));
  REQUIRE(same_values(state.local.edges().data_buffer(),
                      probed.local.edges().data_buffer()));
  REQUIRE(same_values(state.plane_ticket, probed.plane_ticket));
  REQUIRE(same_values(state.group_router, probed.group_router));
  REQUIRE(state.local.defs().size() == probed.local.defs().size());
  for (std::size_t row = 0; row < state.local.defs().size(); ++row)
    REQUIRE(defs_equal(state.local.defs()[row], probed.local.defs()[row]));
}

TEMPLATE_TEST_CASE("plane wave: no wave, no work",
                   "[cut][planes][canonicalize]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // On a virgin state the three operations allocate nothing at all; on a
  // populated one they are byte-stable, and running them twice changes nothing
  // the second time.
  const auto world = make_world<Index, Int>();
  const auto none = make_wave<Index>({}, {0}, {}, 0);
  state_t<Index, Int> virgin;
  products_t<Index> virgin_out;
  REQUIRE(run_wave(
      world, virgin, none, [](Index face) { return plane_of_face(face); },
      virgin_out));
  REQUIRE(allocates_nothing(virgin, virgin_out));

  state_t<Index, Int> state;
  products_t<Index> out;
  REQUIRE(port(world, state, {0, 1}));
  REQUIRE(run_wave(
      world, state, make_wave<Index>({0, 1}, {0, 1, 2}, {5, 5}, 6),
      [](Index face) { return plane_of_face(face); }, out));

  const auto defs = state.local.defs().size();
  const auto groups = state.local.n_canon();
  const auto rows = state.local.edges().data_buffer().size();
  const tf::buffer<Index> ticket = state.plane_ticket;
  const tf::buffer<Index> router = state.group_router;
  products_t<Index> again;
  REQUIRE(run_wave(
      world, state, none, [](Index face) { return plane_of_face(face); },
      again));
  REQUIRE(state.local.defs().size() == defs);
  REQUIRE(state.local.n_canon() == groups);
  REQUIRE(state.local.edges().data_buffer().size() == rows);
  REQUIRE(same_values(ticket, state.plane_ticket));
  REQUIRE(same_values(router, state.group_router));
  REQUIRE(again.layout.size() == 0);
  REQUIRE(again.proposals.size() == 0);
  REQUIRE(again.probe.size() == 0);
  REQUIRE(again.final_row.size() == 0);
  REQUIRE(again.losers.size() == 0);
}

TEMPLATE_TEST_CASE("plane wave: two losers in one non-diff carrier",
                   "[cut][planes][canonicalize]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // Both parents fuse into standing LOCAL groups whose only other carrier is
  // plane 2, which no rebuild reaches: its block keeps its ticket and its
  // order, every retired row becomes the winner's row of the same provenance,
  // and its inert rows are untouched. Each winner's span carries BOTH parents
  // of the fusion — the child's instance and the standing one it superseded.
  constexpr std::size_t inert_groups = 8190;
  const auto world = make_two_loser_world<Index, Int>(inert_groups);
  const auto group_base = Index(4 + inert_groups);
  state_t<Index, Int> state;
  products_t<Index> out;
  REQUIRE(port(world, state, {0, 1, 2}));
  REQUIRE(run_wave(
      world, state, make_wave<Index>({2, 3}, {0, 1, 2}, {0, 1}, 2),
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(values_are<Index>(out.probe, {0, 1}));
  REQUIRE(out.losers.size() == 2);
  REQUIRE(out.losers[0][0] == Index(0));
  REQUIRE(out.losers[0][1] == group_base);
  REQUIRE(out.losers[1][0] == Index(1));
  REQUIRE(out.losers[1][1] == group_base + Index(2));
  REQUIRE(state.local.n_canon() == group_base + Index(4));
  REQUIRE(group_faces_are(state, int(group_base), {0, 2}));
  REQUIRE(group_faces_are(state, int(group_base) + 2, {1, 2}));
  REQUIRE(values_are<Index>(state.plane_ticket, {3, 4, 2}));
  REQUIRE(state.group_router.size() == std::size_t(group_base));
  REQUIRE(state.group_router[0] == group_base);
  REQUIRE(state.group_router[1] == group_base + Index(2));
  REQUIRE(state.group_router[2] == Index(-2));
  REQUIRE(state.group_router[3] == Index(-2));

  std::size_t first_winner = 0;
  std::size_t second_winner = 0;
  std::size_t inert = 0;
  std::size_t retired = 0;
  const auto block_rows = state.local.plane_edges(state.plane_ticket[2]);
  REQUIRE(block_rows.size() == inert_groups + 2);
  for (const auto row : block_rows) {
    const auto &def = state.local.defs()[std::size_t(row)];
    first_winner += std::size_t(def.id == group_base);
    second_winner += std::size_t(def.id == group_base + Index(2));
    inert += std::size_t(def.id >= Index(4) && def.id < group_base);
    retired += std::size_t(def.id == Index(0) || def.id == Index(1));
  }
  REQUIRE(retired == 0);
  REQUIRE(first_winner == 1);
  REQUIRE(second_winner == 1);
  REQUIRE(inert == inert_groups);
  REQUIRE(blocks_are_key_ordered(state));
}

TEMPLATE_TEST_CASE("plane wave: two moved keys meet, a third meets nothing",
                   "[cut][planes][canonicalize][weld]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // Vertices 2 and 3 are both absorbed by created identity 6, so the two
  // groups naming them leave the surgery holding ONE key and the third leaves
  // holding a key of its own. The pair fuses — one mint, the union of both
  // instances, both parents retiring structurally in carriers no rebuild
  // reaches — while the lone group keeps its identity and mints nothing, which
  // is what makes the fusion's mint `base + 0` and not `base + 1`. The probe is
  // EMPTY: no standing group can hold either key, so `changed_groups`, not the
  // sweep, is what states a weld participant.
  const auto world = make_weld_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  weld_t<Index> weld;
  const tf::buffer<Index> no_targets;
  REQUIRE(port(world, state, {1, 2, 4}));
  REQUIRE(values_are<Index>(state.plane_ticket, {-1, 0, 1, -1, 2}));
  REQUIRE(values_are<Index>(state.group_router, {-1, -1, 0, 1, 2}));
  state_merges(weld, {{2, 6}, {3, 6}});
  REQUIRE(run_welded_wave(
      world, state, make_wave<Index>({}, {0}, {}, 0), weld, no_targets,
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(values_are<Index>(weld.row_offsets, {0, 0, 1, 2, 2, 3}));
  REQUIRE(local_tickets_are(world, weld.row_groups, {1, 2, 0}));
  REQUIRE(values_are<Index>(weld.changed, {0, 1, 2}));
  REQUIRE(weld.recdt_planes.size() == 0);
  REQUIRE(out.probe.size() == 0);
  REQUIRE(state.local.defs().size() == 5);
  REQUIRE(state.local.n_canon() == Index(4));
  REQUIRE(values_are<Index>(state.local.def_offsets(), {0, 1, 2, 3, 5}));
  REQUIRE(losers_are(out, {{1, 3}, {2, 3}}));
  REQUIRE(group_faces_are(state, 3, {1, 2}));
  REQUIRE(group_key_is(state, 0, {-1, 6, 0, 0}));
  REQUIRE(values_are<Index>(state.plane_ticket, {-1, 0, 1, -1, 2}));
  REQUIRE(values_are<Index>(state.group_router, {-1, -1, 0, 3, 3}));
  REQUIRE(block_is(state, 1, {3}));
  REQUIRE(block_is(state, 2, {4}));
  REQUIRE(block_is(state, 4, {0}));
  REQUIRE(blocks_are_key_ordered(state));

  const auto triangles = make_triples<Index>(
      {{0, 1, 2}, {10, 1, 0}, {10, 1, 0}, {0, 1, 2}, {10, 0, 1}});
  auto slots = make_triples<Index>(
      {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[1][0] == Index(8));
  REQUIRE(slots[2][0] == Index(8));
  REQUIRE(slots[4][0] == Index(5));
  REQUIRE(span_faces_are(definitions, 3, {1, 2}));
  REQUIRE(span_faces_are(definitions, 0, {4}));
}

TEMPLATE_TEST_CASE("plane wave: a moved key meets a split child",
                   "[cut][planes][canonicalize][weld]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // The parent `(0,0)-(0,1)` is cut at the created identity `7` this wave
  // minted, and that very identity absorbs vertex 2 — so the welded group and
  // the parent's second child are one key. A fresh identity appears in no
  // standing row, so the probe is EMPTY and the sweep never runs: the two
  // statements meet in the sort alone. They fuse into one mint whose span
  // carries BOTH — the child's instance off face 0 and the welded instance off
  // face 1. The parent's carrier rebuilds; the weld participant's carrier,
  // which no rebuild reaches, is substituted in place.
  const auto world = make_weld_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  weld_t<Index> weld;
  const tf::buffer<Index> no_targets;
  REQUIRE(port(world, state, {0, 1}));
  REQUIRE(values_are<Index>(state.plane_ticket, {0, 1, -1, -1, -1}));
  REQUIRE(values_are<Index>(state.group_router, {-1, 0, -1, 1, -1}));
  state_merges(weld, {{2, 7}});
  REQUIRE(run_welded_wave(
      world, state, make_wave<Index>({0}, {0, 1}, {7}, 7), weld, no_targets,
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(local_tickets_are(world, weld.row_groups, {1}));
  REQUIRE(values_are<Index>(weld.changed, {1}));
  REQUIRE(weld.recdt_planes.size() == 0);
  REQUIRE(out.probe.size() == 0);
  REQUIRE(state.local.defs().size() == 5);
  REQUIRE(state.local.n_canon() == Index(4));
  REQUIRE(values_are<Index>(state.local.def_offsets(), {0, 1, 2, 3, 5}));
  REQUIRE(values_are<Index>(out.final_row, {2, 3}));
  REQUIRE(losers_are(out, {{1, 3}}));
  REQUIRE(group_faces_are(state, 2, {0}));
  REQUIRE(group_faces_are(state, 3, {0, 1}));
  REQUIRE(values_are<Index>(state.plane_ticket, {2, 1, -1, -1, -1}));
  REQUIRE(values_are<Index>(state.group_router, {-1, -2, -1, 3, -1}));
  REQUIRE(block_is(state, 0, {2, 3}));
  REQUIRE(block_is(state, 1, {4}));
  REQUIRE(blocks_are_key_ordered(state));

  const auto triangles = make_triples<Index>(
      {{11, 0, 1}, {11, 1, 0}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}});
  auto slots = make_triples<Index>(
      {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[0][0] == Index(7));
  REQUIRE(slots[1][0] == Index(8));
  REQUIRE(span_faces_are(definitions, 3, {0, 1}));
}

TEMPLATE_TEST_CASE("plane wave: a moved key meets an untouched world group",
                   "[cut][planes][canonicalize][weld]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // Vertex 2 is absorbed by the created identity `0` that the immutable group
  // on face 3 already names, so the welded group's new key is that group's
  // key. The sweep finds the world row through the merge TARGET in the probe,
  // and the class is a fusion even though its rows are one changed group's.
  // THE OWNERSHIP LAW promoted the untouched carrier before the port, so the
  // root's group is a standing LOCAL participant: the winner mints from both
  // spans, both locals lose, and the router COMPOSES the root onto the winner.
  // Both carriers answer with ONE ticket.
  const auto world = make_weld_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  weld_t<Index> weld;
  REQUIRE(port(world, state, {1, 3}));
  REQUIRE(values_are<Index>(state.plane_ticket, {-1, 0, -1, 1, -1}));
  REQUIRE(values_are<Index>(state.group_router, {0, -1, -1, 1, -1}));
  state_merges(weld, {{2, 0}});
  REQUIRE(run_welded_wave(
      world, state, make_wave<Index>({}, {0}, {}, 0), weld, weld.targets,
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(values_are<Index>(weld.changed, {1}));
  REQUIRE(values_are<Index>(out.probe, {0}));
  REQUIRE(state.local.defs().size() == 4);
  REQUIRE(state.local.n_canon() == Index(3));
  REQUIRE(values_are<Index>(state.local.def_offsets(), {0, 1, 2, 4}));
  REQUIRE(losers_are(out, {{0, 2}, {1, 2}}));
  REQUIRE(group_faces_are(state, 2, {1, 3}));
  REQUIRE(values_are<Index>(state.plane_ticket, {-1, 0, -1, 1, -1}));
  REQUIRE(values_are<Index>(state.group_router, {2, -1, -1, 2, -1}));
  REQUIRE(block_is(state, 1, {2}));
  REQUIRE(block_is(state, 3, {3}));
  REQUIRE(blocks_are_key_ordered(state));

  const auto triangles = make_triples<Index>(
      {{0, 1, 2}, {4, 1, 0}, {0, 1, 2}, {4, 1, 0}, {0, 1, 2}});
  auto slots = make_triples<Index>(
      {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {0, -1, -1}, {-1, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[1][0] == Index(7));
  REQUIRE(slots[3][0] == Index(7));
  REQUIRE(span_faces_are(definitions, 2, {1, 3}));
}

TEMPLATE_TEST_CASE("plane wave: a moved key that meets nothing mints nothing",
                   "[cut][planes][canonicalize][weld]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // Vertex 2 is absorbed by a created identity no other group names, so the
  // changed group's class holds its instances alone. It already changed in
  // place, so it MINTS NOTHING and keeps its id: no group, no definition row,
  // no loser, no router write, no block move. The probe carries the target, so
  // the sweep names the group a second time — and the participants are one
  // set, so it enters once.
  const auto world = make_weld_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  weld_t<Index> weld;
  REQUIRE(port(world, state, {1}));
  state_merges(weld, {{2, 5}});
  REQUIRE(run_welded_wave(
      world, state, make_wave<Index>({}, {0}, {}, 0), weld, weld.targets,
      [](Index face) { return plane_of_face(face); }, out));

  REQUIRE(values_are<Index>(weld.changed, {0}));
  REQUIRE(values_are<Index>(out.probe, {5}));
  REQUIRE(state.local.defs().size() == 1);
  REQUIRE(state.local.n_canon() == Index(1));
  REQUIRE(values_are<Index>(state.local.def_offsets(), {0, 1}));
  REQUIRE(out.losers.size() == 0);
  REQUIRE(out.final_row.size() == 0);
  REQUIRE(group_key_is(state, 0, {-1, 5, 0, 1}));
  REQUIRE(group_faces_are(state, 0, {1}));
  REQUIRE(values_are<Index>(state.plane_ticket, {-1, 0, -1, -1, -1}));
  REQUIRE(values_are<Index>(state.group_router, {-1, -1, -1, 0, -1}));
  REQUIRE(state.local.edges().size() == 1);
  REQUIRE(state.local.edges().data_buffer().size() == 1);
  REQUIRE(block_is(state, 1, {0}));
  REQUIRE(blocks_are_key_ordered(state));

  const auto triangles = make_triples<Index>(
      {{0, 1, 2}, {9, 1, 0}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}});
  auto slots = make_triples<Index>(
      {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[1][0] == Index(5));
  REQUIRE(span_faces_are(definitions, 0, {1}));
}

TEMPLATE_TEST_CASE("plane wave: a split-only wave is the wave it always was",
                   "[cut][planes][canonicalize][weld]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // One world and one wave, run twice: once through the three operations
  // alone, once through the weld path with an EMPTY delta — the gather
  // publishing no row, the surgery answering with no changed group, the
  // canonicalize stating that empty set. Every buffer either run leaves, and
  // everything either publishes, is equal.
  const auto world = make_world<Index, Int>();
  state_t<Index, Int> state;
  products_t<Index> out;
  REQUIRE(port(world, state, {0, 2, 3}));
  REQUIRE(run_wave(
      world, state, make_wave<Index>({1}, {0, 1}, {0}, 1),
      [](Index face) { return plane_of_face(face); }, out));

  state_t<Index, Int> welded;
  products_t<Index> welded_out;
  weld_t<Index> weld;
  const tf::buffer<Index> no_targets;
  REQUIRE(port(world, welded, {0, 2, 3}));
  REQUIRE(run_welded_wave(
      world, welded, make_wave<Index>({1}, {0, 1}, {0}, 1), weld, no_targets,
      [](Index face) { return plane_of_face(face); }, welded_out));
  REQUIRE(weld.row_offsets.size() == 0);
  REQUIRE(weld.changed.size() == 0);
  REQUIRE(weld.recdt_planes.size() == 0);
  REQUIRE(runs_match(state, out, welded, welded_out));

  REQUIRE(state.local.defs().size() == 7);
  REQUIRE(state.local.n_canon() == Index(4));
  REQUIRE(losers_are(out, {{0, 2}}));
  REQUIRE(values_are<Index>(out.final_row, {3, 6}));
  REQUIRE(values_are<Index>(state.group_router, {2, -2, -1}));
  REQUIRE(values_are<Index>(state.plane_ticket, {3, -1, 1, 2}));
  REQUIRE(blocks_are_key_ordered(state));
}

TEMPLATE_TEST_CASE(
    "plane wave: an unchanged group stays the world's for both carriers",
    "[cut][planes][canonicalize][ownership]", tf::exact::int32,
    tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // THE OWNERSHIP LAW's second clause, on its own: group 1 changes and its
  // only carrier — plane 0 — is ported for it, while group 0, which plane 0
  // and plane 1 share, changes nothing. AN UNCHANGED GROUP STAYS THE WORLD'S,
  // VERBATIM, FOR BOTH CARRIERS: it is never routed, plane 1 is never dragged
  // into the wave, the ported carrier keeps naming the world's own row for it,
  // and the ticket both carriers publish is the immutable root's.
  const auto world = make_shared_bystander_world<Index, Int>();
  state_t<Index, Int> state;
  tf::buffer<Index> taken;
  taken.push_back(Index(1));
  REQUIRE(port_taking(world, state, {0}, taken));

  REQUIRE(values_are<Index>(state.group_router, {-1, 0}));
  REQUIRE(values_are<Index>(state.plane_ticket, {0, -1}));
  // the shared group is named by the world's own row, carried; the taken one
  // by this arrangement's
  REQUIRE(block_is(state, 0, {-1, 0}));
  REQUIRE(state.local.n_canon() == Index(1));
  REQUIRE(state.local.defs().size() == 1);

  // one triangle per plane, each naming the shared group's key; the carrier
  // the wave never touched carries the ticket its emission wrote
  const auto triangles = make_triples<Index>({{4, 0, 1}, {4, 0, 1}});
  auto slots = make_triples<Index>({{-1, -1, -1}, {0, -1, -1}});
  tf::offset_block_buffer<Index, canonicalize_def_t<Index>> definitions;
  REQUIRE(publish(world, state, triangles, slots, definitions));
  REQUIRE(slots[0][0] == Index(0));
  REQUIRE(slots[1][0] == Index(0));
  // and the PA-owned suffix holds the taken group alone
  REQUIRE(definitions.size() == 1);
  REQUIRE(span_faces_are(definitions, 0, {0}));
}

// ============================================================================
// THE RIDER LAW: a promoted face's shared side joins the group it names.
// ============================================================================
//
// A face the cut world never named gets A NEW SPAN for every edge that is its
// alone. The ONE row that cannot go there is the SHARED edge: its canonical
// group already exists — the neighbour's instance and the wave's split live
// on it, the group IS the cross-face join, and a second group for one wall is
// the twin-wall defect. So that row JOINS the group, and the port is where it
// lands.
//
// The two cases below are one differential: the same world, the same taken
// group, ported WITHOUT and WITH the rider. What may not move between them is
// every world row's rank, because the port's map IS that rank and no
// correspondence table exists.
TEMPLATE_TEST_CASE("plane wave: a rider joins the group the port takes",
                   "[cut][planes][port][entrance]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // group 1 is `AB (0,0)-(0,1)`, carried by face 0 ALONE — the shape a shared
  // edge has when the face across it is one the cut world never named
  const auto world = make_world<Index, Int>();
  tf::buffer<Index> taken;
  fill_buffer(taken, {1});

  state_t<Index, Int> plain;
  REQUIRE(port_taking(world, plain, {0}, taken));
  REQUIRE(plain.local.n_canon() == Index(1));
  REQUIRE(values_are<Index>(plain.local.def_offsets(), {0, 1}));
  REQUIRE(group_faces_are(plain, 0, {0}));
  REQUIRE(block_is(plain, 0, {0}));

  // the rider states the SAME key from a face stamped past every face the
  // world holds, which is what the promotion stamps an entrant with
  tf::buffer<canonicalize_def_t<Index>> riders;
  riders.push_back(make_def<Index>(0, 0, 0, 1, Index(-1), Index(4), 7, 0));
  state_t<Index, Int> ridden;
  REQUIRE(port_riding(world, ridden, {0}, taken, riders, {0, 1}));

  // ONE group, TWO instances: the world's own, then the rider
  REQUIRE(ridden.local.n_canon() == Index(1));
  REQUIRE(values_are<Index>(ridden.local.def_offsets(), {0, 2}));
  REQUIRE(group_faces_are(ridden, 0, {0, 4}));
  REQUIRE(defs_equal(ridden.local.defs()[0], plain.local.defs()[0]));
  REQUIRE(ridden.local.defs()[1].id == Index(0));
  REQUIRE(ridden.local.defs()[1].face == Index(4));
  // THE MAP IS THE RANK, AND THE RIDER DOES NOT MOVE IT: the carrier the port
  // rewrote names the same row it named without the rider
  REQUIRE(block_is(ridden, 0, {0}));
  REQUIRE(same_values(ridden.plane_ticket, plain.plane_ticket));
  REQUIRE(same_values(ridden.group_router, plain.group_router));
  REQUIRE(blocks_are_key_ordered(ridden));
}

TEMPLATE_TEST_CASE("plane wave: riders close a span they do not reorder",
                   "[cut][planes][port][entrance]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Index = typename index_of<Int>::type;

  // group 0 is `AX (-1,0)-(0,0)`, carried TWICE — by faces 2 and 3 — so the
  // rank formula has something to preserve
  const auto world = make_world<Index, Int>();
  tf::buffer<Index> taken;
  fill_buffer(taken, {0});

  state_t<Index, Int> plain;
  REQUIRE(port_taking(world, plain, {2, 3}, taken));
  REQUIRE(values_are<Index>(plain.local.def_offsets(), {0, 2}));
  REQUIRE(block_is(plain, 2, {0}));
  REQUIRE(block_is(plain, 3, {1}));

  tf::buffer<canonicalize_def_t<Index>> riders;
  riders.push_back(make_def<Index>(-1, 0, 0, 0, Index(-1), Index(4), 7, 0));
  riders.push_back(make_def<Index>(-1, 0, 0, 0, Index(-1), Index(5), 8, 0));
  state_t<Index, Int> ridden;
  REQUIRE(port_riding(world, ridden, {2, 3}, taken, riders, {0, 2}));

  REQUIRE(ridden.local.n_canon() == Index(1));
  REQUIRE(values_are<Index>(ridden.local.def_offsets(), {0, 4}));
  // the world's instances lead in their own order; the riders close the span
  // in theirs, because a stamp past every world face sorts last
  REQUIRE(group_faces_are(ridden, 0, {2, 3, 4, 5}));
  REQUIRE(defs_equal(ridden.local.defs()[0], plain.local.defs()[0]));
  REQUIRE(defs_equal(ridden.local.defs()[1], plain.local.defs()[1]));
  REQUIRE(block_is(ridden, 2, {0}));
  REQUIRE(block_is(ridden, 3, {1}));
  REQUIRE(blocks_are_key_ordered(ridden));
}
