/*
 * Copyright (c) 2025 XLAB
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
#include "../core/algorithm/circular_increment.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/intersects.hpp"
#include "../core/local_value.hpp"
#include "../core/small_vector.hpp"
#include "../exact/triangle_segment_intersection.hpp"
#include "../exact/vertex.hpp"
#include "../exact/vertex_converter.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./exact/coplanar_primitives.hpp"
#include "./exact/crossing_edges_vs_face.hpp"
#include "./exact/dedup_coincident_points.hpp"
#include "./exact/duplicate_tagged_intersection.hpp"
#include "./exact/face_plane_info.hpp"
#include "./exact/make_kernel.hpp"
#include "./exact/predicate_kernel.hpp"
#include "./exact/tagged_intersections.hpp"
#include "./exact/vertex_face.hpp"
#include "./impl/face_pair_search.hpp"
#include "./intersect_config.hpp"

namespace tf {

template <typename Index, typename RealType, typename Int = tf::exact::int32>
class intersections_within_polygons
    : public tf::intersect::tagged_intersections<Index, Int, 3> {
  static constexpr std::size_t Dims = 3;
  using base_t = tf::intersect::tagged_intersections<Index, Int, Dims>;
  using intersection_t = tf::intersect::tagged_intersection<Index>;

public:
  auto converter() const -> const auto & { return _converter; }

  template <typename Policy>
  auto build(const tf::polygons<Policy> &form,
             tf::intersect_config config = {}) {
    static_assert(tf::has_tree_policy<Policy>, "Use polygons | tf::tag(tree)");
    static_assert(tf::has_manifold_edge_link_policy<Policy>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_face_membership_policy<Policy>,
                  "Use polygons | tf::tag(face_membership)");

    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(form);

    if (config.mode & tf::intersect_mode::primitives)
      build_primitives(form, tf::exact::make_kernel(_converter, config.tolerance));
    else
      build_sos(form);
  }

private:

  template <typename Indices0, typename Indices1>
  static auto compute_shared_masks(const Indices0 &ids0, const Indices1 &ids1,
                                   tf::buffer<bool> &s0, tf::buffer<bool> &s1) {
    s0.allocate(ids0.size());
    s1.allocate(ids1.size());
    std::fill(s0.begin(), s0.end(), false);
    std::fill(s1.begin(), s1.end(), false);
    for (decltype(ids0.size()) i = 0; i < ids0.size(); ++i)
      for (decltype(ids1.size()) j = 0; j < ids1.size(); ++j)
        if (Index(ids0[i]) == Index(ids1[j]))
          s0[i] = s1[j] = true;
  }

  static auto has_shared_edge(const tf::buffer<bool> &mask) -> bool {
    std::size_t n = mask.size();
    for (std::size_t i = 0; i < n; ++i)
      if (mask[i] && mask[tf::circular_increment(i, n)])
        return true;
    return false;
  }

  static auto edge_vs_convex_face_sos(tf::exact::vertex_range<Index, Int> face,
                                      const tf::exact::vertex<Index, Int> &v0,
                                      const tf::exact::vertex<Index, Int> &v1)
      -> std::optional<tf::exact::pt3<Int>> {
    auto n = face.size();
    for (std::size_t t = 0; t + 2 < n; ++t) {
      if (auto pt = tf::exact::triangle_segment_intersect_point_sos(
              std::array<tf::exact::vertex<Index, Int>, 5>{
                  face[0], face[t + 1], face[t + 2], v0, v1}))
        return pt;
    }
    return std::nullopt;
  }

  /// Edges of one prepped self face vs the other's face; skips edges with a
  /// shared endpoint (a shared edge does not self-intersect).
  template <typename EdgeIsRep, typename Ints, typename Pts>
  static auto edges_vs_face_sos(tf::exact::vertex_range<Index, Int> edge_verts,
                                Index edge_id,
                                tf::exact::vertex_range<Index, Int> face_verts,
                                Index face_id, const EdgeIsRep &edge_is_rep,
                                const tf::buffer<bool> &shared, Ints &ints,
                                Pts &pts) {
    auto n = edge_verts.size();
    for (std::size_t j = 0; j < n; ++j) {
      auto next_j = tf::circular_increment(j, n);
      if (shared[j] || shared[next_j])
        continue;
      if (!edge_is_rep(j))
        continue;
      if (auto pt = edge_vs_convex_face_sos(face_verts, edge_verts[j],
                                            edge_verts[next_j])) {
        Index id = pts.size();
        pts.push_back(*pt);
        ints.push_back({Index(0), Index(0), edge_id, face_id,
                        {Index(j), tf::topo_type::edge},
                        {face_id, tf::topo_type::face}, id});
      }
    }
  }

  template <typename Form, typename MEL>
  static void
  self_sos_process(tf::intersect::face_pair_workspace<Index, Int> &ws,
                   bool is_self, const Form &form, const MEL &mel) {
    auto n0 = ws.n0();
    auto n1 = ws.n1();
    for (std::size_t i = 0; i < n0; ++i)
      for (std::size_t j = (i + 1) * is_self; j < n1; ++j) {
        if (!tf::intersects(ws.ibox0[i], ws.ibox1[j]))
          continue;
        auto id0 = Index(ws.ids0[i]);
        auto id1 = Index(ws.ids1[j]);
        compute_shared_masks(form[ws.ids0[i]].indices(),
                             form[ws.ids1[j]].indices(), ws.shared0, ws.shared1);
        if (has_shared_edge(ws.shared0))
          continue;
        auto erep0 = [&](std::size_t e) {
          return mel[id0][e].is_representative(id0);
        };
        auto erep1 = [&](std::size_t e) {
          return mel[id1][e].is_representative(id1);
        };
        edges_vs_face_sos(ws.face0(i), id0, ws.face1(j), id1, erep0, ws.shared0,
                          ws.intersections, ws.points);
        edges_vs_face_sos(ws.face1(j), id1, ws.face0(i), id0, erep1, ws.shared1,
                          ws.intersections, ws.points);
      }
  }

  template <typename Policy>
  auto run_self_sos_pair(
      const tf::polygons<Policy> &form,
      tf::local_value<tf::intersect::face_pair_workspace<Index, Int>> &ws) {
    auto &&mel = form.manifold_edge_link();
    tf::intersect::search_face_pairs_self(
        form, _converter, Int(0), ws,
        [&](tf::intersect::face_pair_workspace<Index, Int> &w, bool is_self) {
          self_sos_process(w, is_self, form, mel);
        });
  }

  /// Self pair test over two prepped faces (ranges + cached planes + shared
  /// masks). Shared-aware masks, `_self` crossing dedup, both-sided reject.
  template <typename Poly0, typename Poly1, typename MEL, typename FM,
            typename Ints, typename Pts>
  static auto within_polygon_pair_prepped(
      const Poly0 &poly0, const Poly1 &poly1, const MEL &mel, const FM &fm,
      tf::exact::vertex_range<Index, Int> face_buf0,
      const tf::exact::face_plane<Int> &fp0,
      tf::exact::vertex_range<Index, Int> face_buf1,
      const tf::exact::face_plane<Int> &fp1, const tf::buffer<bool> &shared0,
      const tf::buffer<bool> &shared1, Ints &ints, Pts &pts,
      const tf::exact::predicate_kernel<Int> &kernel = {}) {
    auto face0_id = Index(poly0.id());
    auto face1_id = Index(poly1.id());

    auto n0 = face_buf0.size();
    auto n1 = face_buf1.size();

    const auto &plane0 = fp0.info;
    const auto &plane1 = fp1.info;

    constexpr int has_negative = 1 << 0;
    constexpr int has_zero = 1 << 1;
    constexpr int has_positive = 1 << 2;
    constexpr int has_crossing = has_negative | has_positive;

    tf::small_vector<int, 16> signs0, signs1;
    signs0.resize(n0);
    signs1.resize(n1);
    int mask0 = 0, mask1 = 0;
    int mask0_ns = 0, mask1_ns = 0;
    if (plane1.valid)
      for (decltype(n0) i = 0; i < n0; ++i) {
        signs0[i] = tf::exact::orient3d_plane_sign(fp1.plane, face_buf0[i].pt);
        auto bit = 1 << (signs0[i] + 1);
        mask0 |= bit;
        if (!shared0[i])
          mask0_ns |= bit;
      }
    if (plane0.valid)
      for (decltype(n1) j = 0; j < n1; ++j) {
        signs1[j] = tf::exact::orient3d_plane_sign(fp0.plane, face_buf1[j].pt);
        auto bit = 1 << (signs1[j] + 1);
        mask1 |= bit;
        if (!shared1[j])
          mask1_ns |= bit;
      }

    int combined = mask0 | mask1;
    if (!(combined & has_zero) && (mask0 & has_crossing) != has_crossing &&
        (mask1 & has_crossing) != has_crossing)
      return;

    int combined_ns = mask0_ns | mask1_ns;
    if (!(combined_ns & has_zero) &&
        (mask0_ns & has_crossing) != has_crossing &&
        (mask1_ns & has_crossing) != has_crossing)
      return;

    bool any_zero = combined & has_zero;
    auto is_rep0 = [&](std::size_t i) -> std::pair<bool, bool> {
      auto v_global = poly0.indices()[i];
      return {Index(fm[v_global].front()) == face0_id,
              mel[face0_id][i].is_representative(face0_id)};
    };
    auto is_rep1 = [&](std::size_t i) -> std::pair<bool, bool> {
      auto v_global = poly1.indices()[i];
      return {Index(fm[v_global].front()) == face1_id,
              mel[face1_id][i].is_representative(face1_id)};
    };

    auto shared0_f = [&shared0](std::size_t i) { return shared0[i]; };
    auto shared1_f = [&shared1](std::size_t j) { return shared1[j]; };

    bool both_crossing = (mask0 & has_crossing) == has_crossing &&
                         (mask1 & has_crossing) == has_crossing;
    if ((mask0 & has_crossing) == has_crossing)
      tf::exact::crossing_edges_vs_face_self(
          face_buf0, n0, face_buf1, n1, signs0, 0, 0, face0_id, face1_id,
          is_rep0, is_rep1, ints, pts, both_crossing, kernel);
    if ((mask1 & has_crossing) == has_crossing)
      tf::exact::crossing_edges_vs_face_self(
          face_buf1, n1, face_buf0, n0, signs1, 0, 0, face1_id, face0_id,
          is_rep1, is_rep0, ints, pts, both_crossing, kernel);
    if (!any_zero)
      return;

    tf::exact::coplanar_primitives(face_buf0, n0, face_buf1, n1, signs0, signs1,
                                   0, 0, face0_id, face1_id, is_rep0, is_rep1,
                                   plane0, plane1, shared0_f, shared1_f, ints,
                                   pts, kernel);

    auto vrep0 = [&](std::size_t i) {
      return Index(fm[poly0.indices()[i]].front()) == face0_id;
    };
    auto vrep1 = [&](std::size_t j) {
      return Index(fm[poly1.indices()[j]].front()) == face1_id;
    };
    tf::exact::vertex_face(face_buf0, n0, face_buf1, n1, signs0, 0, 0, face0_id,
                           face1_id, vrep0, shared0_f, plane1, ints, pts,
                           kernel);
    tf::exact::vertex_face(face_buf1, n1, face_buf0, n0, signs1, 0, 0, face1_id,
                           face0_id, vrep1, shared1_f, plane0, ints, pts,
                           kernel);
  }

  /// Per-leaf-pair self primitives logic; on the diagonal leaf (`is_self`) the
  /// inner loop skips the self-pair and each unordered pair's mirror.
  template <typename Form, typename MEL, typename FM>
  static void self_process(tf::intersect::face_pair_workspace<Index, Int> &ws,
                           bool is_self, const Form &form, const MEL &mel,
                           const FM &fm,
                           const tf::exact::predicate_kernel<Int> &kernel) {
    auto n0 = ws.n0();
    auto n1 = ws.n1();
    ws.fp0.allocate(n0);
    for (std::size_t i = 0; i < n0; ++i)
      ws.fp0[i] = tf::exact::make_face_plane(ws.face0(i));
    ws.fp1.allocate(n1);
    for (std::size_t j = 0; j < n1; ++j)
      ws.fp1[j] = tf::exact::make_face_plane(ws.face1(j));
    for (std::size_t i = 0; i < n0; ++i)
      for (std::size_t j = (i + 1) * is_self; j < n1; ++j) {
        if (!tf::intersects(ws.ibox0[i], ws.ibox1[j]))
          continue;
        auto poly0 = tf::tag_id(Index(ws.ids0[i]), form[ws.ids0[i]]);
        auto poly1 = tf::tag_id(Index(ws.ids1[j]), form[ws.ids1[j]]);
        compute_shared_masks(poly0.indices(), poly1.indices(), ws.shared0,
                             ws.shared1);
        within_polygon_pair_prepped(poly0, poly1, mel, fm, ws.face0(i),
                                    ws.fp0[i], ws.face1(j), ws.fp1[j],
                                    ws.shared0, ws.shared1, ws.intersections,
                                    ws.points, kernel);
      }
  }

  template <typename Policy>
  auto run_self_pair(
      const tf::polygons<Policy> &form,
      const tf::exact::predicate_kernel<Int> &kernel,
      tf::local_value<tf::intersect::face_pair_workspace<Index, Int>> &ws) {
    auto &&mel = form.manifold_edge_link();
    auto &&fm = form.face_membership();
    tf::intersect::search_face_pairs_self(
        form, _converter, kernel.tolerance_int(), ws,
        [&](tf::intersect::face_pair_workspace<Index, Int> &w, bool is_self) {
          self_process(w, is_self, form, mel, fm, kernel);
        });
  }

  template <typename Duplicator>
  auto finalize_build(
      tf::local_value<tf::intersect::face_pair_workspace<Index, Int>> &ws,
      Duplicator &&duplicator) {
    tf::buffer<intersection_t> raw;
    tf::buffer<tf::exact::pt3<Int>> points;
    tf::intersect::merge_face_pair_workspaces(ws, raw, points);
    if (points.size() == 0)
      return base_t::finalize(Index(1));
    tf::intersect::dedup_coincident_points(raw, points);
    base_t::_intersection_points = std::move(points);
    tf::generic_generate(raw, base_t::_intersections,
                         std::forward<Duplicator>(duplicator));
    base_t::finalize(Index(1));
  }

  template <typename Policy> auto build_sos(const tf::polygons<Policy> &form) {
    tf::local_value<tf::intersect::face_pair_workspace<Index, Int>> ws;
    run_self_sos_pair(form, ws);
    finalize_build(ws, [&](auto rec, auto &buffer) {
      tf::intersect::duplicate_intersection_self(
          form.faces(), form.face_membership(), form.manifold_edge_link(), rec,
          buffer);
    });
  }

  template <typename Policy>
  auto build_primitives(const tf::polygons<Policy> &form,
                        const tf::exact::predicate_kernel<Int> &kernel = {}) {
    tf::local_value<tf::intersect::face_pair_workspace<Index, Int>> ws;
    run_self_pair(form, kernel, ws);
    finalize_build(ws, [&](auto rec, auto &buffer) {
      tf::intersect::duplicate_intersection_self(
          form.faces(), form.face_membership(), form.manifold_edge_link(), rec,
          buffer);
    });
  }

  tf::exact::vertex_converter<Int, RealType, Dims> _converter;
};

} // namespace tf
