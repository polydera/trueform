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
#include "../../core/array_hash.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/polygon_intersections.hpp"
#include "./classify_tet_pair.hpp"
#include "./land_tet_reach_points.hpp"
#include "./lift_tet_crossings.hpp"
#include "./state_tet_chord_crossings.hpp"
#include "./state_tet_cut_contacts.hpp"
#include "./tet_changed_triangle.hpp"
#include "./tet_slot_holders.hpp"
#include "./wave_tet_lanes.hpp"
#include <array>
#include <utility>

namespace tf::tetmesh {

/// The scratch every step of a drain walks on (@ref cut_tet_pairs), owned
/// by its caller and reused drain to drain: each step's tables in the order
/// the steps run, the wave's among them (@ref wave_tet_lanes).
template <typename Index, typename Int> struct tet_pair_workspace {
  tf::polygons_buffer<Index, Int, 3, 3> form;
  tf::buffer<Index> local;
  tf::buffer<Index> global;
  tf::buffer<Index> form_triangles;
  tf::polygon_intersections<Index, Int, Int> records;
  tf::buffer<typename tf::exact::meta<Int>::param_type> dyadic;
  tf::buffer<tf::exact::pt3<Int>> positions;
  tf::buffer<std::array<Index, 2>> queue;
  tet_pair_kernel_workspace<Index, Int> pair_kernel;
  tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> traced;
  tf::buffer<std::array<Index, 2>> seeds;
  tet_lift_workspace<Index, Int> lift;
  tf::buffer<Index> cut;
  tf::buffer<Index> cut_facets;
  tf::buffer<Index> group;
  tf::buffer<std::array<Index, 3>> edge_points;
  tet_slot_holders<Index> holders;
  tf::buffer<Index> point_of;
  tf::buffer<std::array<Index, 2>> record_pairs;
  tf::buffer<std::array<Index, 2>> no_pairs;
  tf::buffer<Index> record_ids;
  tf::buffer<Index> entered;
  tf::buffer<Index> retired_slots;
  tet_reach_landing_workspace<Index> landing;
  tf::buffer<typename tf::exact::meta<Int>::param_type> side_t;
  tf::buffer<std::size_t> side_order;
  tf::buffer<std::array<Index, 3>> stated_sides;
  tf::buffer<tet_cut_contact<Index>> contacts;
  tf::buffer<std::array<std::size_t, 2>> face_contacts;
  tf::buffer<Index> ids;
  tf::buffer<tet_cut_chord<Index>> cut_chords;
  tf::buffer<std::array<Index, 3>> chord_points;
  tf::buffer<tet_chord_crossing<Index>> crossings;
  tet_chord_crossing_workspace<Index, Int> chord_work;
  tf::buffer<tf::exact::pt3<Int>> proposed;
  tf::buffer<Index> proposal_name;
  tf::buffer<Index> run_a;
  tf::buffer<Index> run_b;
  tf::buffer<Index> unit_runs;
  tf::buffer<Index> unit_members;
  tf::buffer<Index> unit_of_cut;
  tf::buffer<std::array<Index, 2>> grouped;
  tf::buffer<std::array<Index, 2>> neighbour_units;
  tf::buffer<Index> due_units;
  tf::buffer<std::array<Index, 2>> touched;
  tf::buffer<tet_changed_triangle<Index>> changed;
  tf::buffer<std::array<Index, 3>> all_pieces;
  tet_wave_workspace<Index, Int> wave;
};

} // namespace tf::tetmesh
