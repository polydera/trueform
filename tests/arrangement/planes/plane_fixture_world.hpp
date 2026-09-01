/**
 * @file plane_fixture_world.hpp
 * @brief The owning synthetic plane world the hand-built fixtures state
 *
 * The production world is the local arrangement, borrowed. A fixture states
 * its tables by hand instead, so it owns them — the same final identities and
 * CSR shapes, written row by row. Every plane is in the cut prefix; the uncut
 * partition is consequently an empty suffix.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/core/buffer.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/arrangement/planes/plane_world.hpp>
#include <trueform/exact/plane_frame.hpp>
#include <trueform/intersect/graph/face_descriptor.hpp>
#include <trueform/intersect/graph/plane_tables.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::test {

/// The storage of one hand-built world. The optional source tables and the
/// local merge/split carriers state the raw canonical roots from which a
/// local-arrangement build would have produced the final tables.
template <typename Index, typename Int> class plane_fixture_policy {
public:
  using index_type = Index;
  using tables_t = tf::intersect::graph::plane_tables<Index, Int>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;
  using frame_t = tf::exact::plane_frame<Int>;

  auto tables() -> tables_t & { return _tables; }
  auto tables() const -> const tables_t & { return _tables; }
  auto source_tables() -> tables_t & { return _source_tables; }
  auto source_tables() const -> const tables_t & { return _source_tables; }
  auto frames() -> tf::buffer<frame_t> & { return _frames; }
  auto frames() const -> const tf::buffer<frame_t> & { return _frames; }
  auto plane_members_data() -> tf::offset_block_buffer<Index, Index> & {
    return _plane_members;
  }
  auto descriptor_data() -> tf::buffer<descriptor_t> & { return _descriptors; }
  auto descriptor_data() const -> const tf::buffer<descriptor_t> & {
    return _descriptors;
  }
  auto face_planes() -> tf::buffer<Index> & { return _face_planes; }
  auto face_planes() const -> const tf::buffer<Index> & { return _face_planes; }
  auto face_orientations() -> tf::buffer<std::int8_t> & {
    return _face_orientations;
  }
  auto face_orientations() const -> const tf::buffer<std::int8_t> & {
    return _face_orientations;
  }
  auto local_merges() -> tf::buffer<std::array<Index, 3>> & {
    return _local_merges;
  }
  auto local_split_roots() -> tf::buffer<Index> & { return _local_split_roots; }
  auto local_split_survivors() -> tf::offset_block_buffer<Index, Index> & {
    return _local_split_survivors;
  }

  auto n_planes() const -> Index { return Index(_tables.edges().size()); }
  auto n_faces() const -> Index { return Index(_descriptors.size()); }

  auto frame(Index plane) const -> const frame_t & {
    return _frames[std::size_t(plane)];
  }
  auto member_count(Index plane) const -> Index {
    return Index(_plane_members[std::size_t(plane)].size());
  }
  auto member(Index plane, Index position) const -> Index {
    return _plane_members[std::size_t(plane)][std::size_t(position)];
  }
  auto plane_of_face(Index face) const -> Index {
    return _face_planes[std::size_t(face)];
  }
  auto descriptor(Index face) const -> const descriptor_t & {
    return _descriptors[std::size_t(face)];
  }
  auto descriptors() const { return tf::make_range(_descriptors); }
  auto face_orientation(Index face) const -> std::int8_t {
    return _face_orientations[std::size_t(face)];
  }

  auto edge_defs() const { return _tables.edge_defs(); }
  auto n_canon() const -> Index { return _tables.n_canon(); }
  auto canon_group(Index canon) const { return _tables.canon_group(canon); }
  auto plane_edges(Index plane) const { return _tables.plane_edges(plane); }
  /// The fixture states its tier by hand before the build reads it, so the
  /// transition is already behind.
  auto materialized() const -> bool { return true; }
  auto materialize() -> void {}
  auto merges() const { return tf::make_range(_local_merges); }
  auto split_roots() const { return tf::make_range(_local_split_roots); }
  auto split_survivors(std::size_t group) const {
    return _local_split_survivors[group];
  }

private:
  tables_t _tables;
  tables_t _source_tables;
  tf::buffer<frame_t> _frames;
  tf::offset_block_buffer<Index, Index> _plane_members;
  tf::buffer<descriptor_t> _descriptors;
  tf::buffer<Index> _face_planes;
  tf::buffer<std::int8_t> _face_orientations;
  tf::buffer<std::array<Index, 3>> _local_merges;
  tf::buffer<Index> _local_split_roots;
  tf::offset_block_buffer<Index, Index> _local_split_survivors;
};

template <typename Index, typename Int>
using plane_fixture_world =
    tf::arrangement::plane_world<plane_fixture_policy<Index, Int>>;

} // namespace tf::test
