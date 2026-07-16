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
#include "../core/algorithm/generic_generate.hpp"
#include "../core/local_value.hpp"
#include "../core/polygons.hpp"
#include "../exact/vertex_converter.hpp"
#include "../spatial/policy/tree.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./exact/dedup_coincident_points.hpp"
#include "./exact/duplicate_tagged_intersection.hpp"
#include "./exact/make_kernel.hpp"
#include "./exact/tagged_intersections.hpp"
#include "./impl/face_pair_kernels.hpp"
#include "./impl/face_self_kernels.hpp"
#include "./impl/resolve_shared_vertex_ids.hpp"
#include "./intersect_config.hpp"
#include "tbb/task_group.h"

#include <utility>

namespace tf {

/// Exact intersection data of a set of polygon meshes.
///
/// Subsumes the between-forms and within-form builds behind one record
/// space: a record with `tag != tag_other` is a cross-pair intersection,
/// `tag == tag_other` a self-intersection of that form. Cross pairs are
/// always computed; self records are generated per form when the mode
/// carries @ref tf::intersect_mode::self_intersections (callers write
/// `tf::intersect_mode::within`). The single-form build is self-only and
/// forces the bit — it is the only meaning a one-form build has.
///
/// Stores intersection points as int lattice coordinates computed via
/// exact arithmetic (SoS or primitives). No float round-trip — the int
/// points are the primary representation. Supports convex polygon faces
/// of any size. The classification kernels live in
/// `impl/face_pair_kernels.hpp` / `impl/face_self_kernels.hpp`.
template <typename Index, typename RealType, typename Int = tf::exact::int32>
class polygon_intersections
    : public tf::intersect::tagged_intersections<Index, Int, 3> {
  static constexpr std::size_t Dims = 3;
  using base_t = tf::intersect::tagged_intersections<Index, Int, Dims>;
  using intersection_t = tf::intersect::tagged_intersection<Index>;
  using workspace_t = tf::intersect::face_pair_workspace<Index, Int>;

public:
  auto converter() const -> const auto & { return _converter; }

  /// Self-only build of one form (the `within` bit is implied).
  template <typename Policy>
  auto build(const tf::polygons<Policy> &form,
             tf::intersect_config config = {}) {
    assert_form_policies<Policy>();

    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(form);

    tf::local_value<workspace_t> ws;
    const bool primitives = config.mode & tf::intersect_mode::primitives;
    if (primitives)
      run_self_primitives(form, 0,
                          tf::exact::make_kernel(_converter, config.tolerance),
                          ws);
    else
      run_self_sos(form, 0, ws);

    auto apply_to_form = [&form](int, auto &&f) { f(form); };
    tf::buffer<Index> vertex_offsets;
    if (primitives) {
      vertex_offsets.push_back(Index(0));
      vertex_offsets.push_back(Index(form.points().size()));
    }
    finalize_build(ws, apply_to_form, Index(1), true, !primitives,
                   vertex_offsets);
  }

  /// Two forms of possibly different policies. Cross records always;
  /// per-form self records when the mode carries `self_intersections`.
  template <typename Policy0, typename Policy1>
  auto build(const tf::polygons<Policy0> &form0,
             const tf::polygons<Policy1> &form1,
             tf::intersect_config config = {}) {
    assert_form_policies<Policy0>();
    assert_form_policies<Policy1>();

    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(form0, form1);

    const bool primitives = config.mode & tf::intersect_mode::primitives;
    const bool with_self =
        config.mode & tf::intersect_mode::self_intersections;
    auto kernel = tf::exact::make_kernel(_converter, config.tolerance);

    tf::local_value<workspace_t> ws;
    if (primitives) {
      run_primitives_pair(form0, form1, 0, 1, kernel, ws);
      if (with_self) {
        run_self_primitives(form0, 0, kernel, ws);
        run_self_primitives(form1, 1, kernel, ws);
      }
    } else {
      run_sos_pair(form0, form1, 0, 1, ws);
      if (with_self) {
        run_self_sos(form0, 0, ws);
        run_self_sos(form1, 1, ws);
      }
    }

    auto apply_to_form = [&form0, &form1](int tag, auto &&f) {
      if (tag == 0)
        f(form0);
      else
        f(form1);
    };
    tf::buffer<Index> vertex_offsets;
    if (with_self && primitives) {
      vertex_offsets.push_back(Index(0));
      vertex_offsets.push_back(Index(form0.points().size()));
      vertex_offsets.push_back(
          Index(form0.points().size() + form1.points().size()));
    }
    finalize_build(ws, apply_to_form, Index(2), with_self, !primitives,
                   vertex_offsets);
  }

  /// N forms. Cross records for every pair; per-form self records when
  /// the mode carries `self_intersections`. A one-form range is the
  /// self-only build (the bit is implied — nothing else remains).
  template <typename Iterator, std::size_t N>
  auto build(tf::range<Iterator, N> forms, tf::intersect_config config = {}) {
    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(forms);

    const auto n = Index(forms.size());
    const bool primitives = config.mode & tf::intersect_mode::primitives;
    const bool with_self =
        n == 1 || (config.mode & tf::intersect_mode::self_intersections);
    auto kernel = tf::exact::make_kernel(_converter, config.tolerance);

    tf::local_value<workspace_t> ws;
    tbb::task_group tg;
    for (Index i = 0; i < n; ++i)
      for (Index j = i + 1; j < n; ++j)
        tg.run([&, i, j]() {
          if (primitives)
            run_primitives_pair(forms[i], forms[j], int(i), int(j), kernel,
                                ws);
          else
            run_sos_pair(forms[i], forms[j], int(i), int(j), ws);
        });
    if (with_self)
      for (Index k = 0; k < n; ++k)
        tg.run([&, k]() {
          if (primitives)
            run_self_primitives(forms[k], int(k), kernel, ws);
          else
            run_self_sos(forms[k], int(k), ws);
        });
    tg.wait();

    auto apply_to_form = [forms](int tag, auto &&f) { f(forms[tag]); };
    tf::buffer<Index> vertex_offsets;
    if (with_self && primitives) {
      vertex_offsets.push_back(Index(0));
      for (Index k = 0; k < n; ++k)
        vertex_offsets.push_back(vertex_offsets[std::size_t(k)] +
                                 Index(forms[k].points().size()));
    }
    finalize_build(ws, apply_to_form, n, with_self, !primitives,
                   vertex_offsets);
  }

private:
  template <typename Policy> static constexpr auto assert_form_policies() {
    static_assert(tf::has_tree_policy<Policy>, "Use polygons | tf::tag(tree)");
    static_assert(tf::has_manifold_edge_link_policy<Policy>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_face_membership_policy<Policy>,
                  "Use polygons | tf::tag(face_membership)");
  }

  template <typename Policy0, typename Policy1>
  auto run_sos_pair(const tf::polygons<Policy0> &form0,
                    const tf::polygons<Policy1> &form1, int tag0, int tag1,
                    tf::local_value<workspace_t> &ws) {
    auto &&mel0 = form0.manifold_edge_link();
    auto &&mel1 = form1.manifold_edge_link();
    tf::intersect::search_face_pairs(
        form0, form1, tag0, tag1, _converter, Int(0), ws,
        [&, tag0, tag1](workspace_t &w) {
          tf::intersect::sos_process(w, tag0, tag1, mel0, mel1);
        });
  }

  template <typename Policy0, typename Policy1>
  auto run_primitives_pair(const tf::polygons<Policy0> &form0,
                           const tf::polygons<Policy1> &form1, int tag0,
                           int tag1,
                           const tf::exact::predicate_kernel<Int> &kernel,
                           tf::local_value<workspace_t> &ws) {
    auto &&mel0 = form0.manifold_edge_link();
    auto &&mel1 = form1.manifold_edge_link();
    auto &&fm0 = form0.face_membership();
    auto &&fm1 = form1.face_membership();
    tf::intersect::search_face_pairs(
        form0, form1, tag0, tag1, _converter, kernel.tolerance_int(), ws,
        [&, tag0, tag1](workspace_t &w) {
          tf::intersect::primitives_process(w, form0, form1, tag0, tag1, mel0,
                                            mel1, fm0, fm1, kernel);
        });
  }

  template <typename Policy>
  auto run_self_sos(const tf::polygons<Policy> &form, int tag,
                    tf::local_value<workspace_t> &ws) {
    auto &&mel = form.manifold_edge_link();
    tf::intersect::search_face_pairs_self(
        form, _converter, Int(0), ws, [&, tag](workspace_t &w, bool is_self) {
          tf::intersect::self_sos_process(w, is_self, form, tag, mel);
        });
  }

  template <typename Policy>
  auto run_self_primitives(const tf::polygons<Policy> &form, int tag,
                           const tf::exact::predicate_kernel<Int> &kernel,
                           tf::local_value<workspace_t> &ws) {
    auto &&mel = form.manifold_edge_link();
    auto &&fm = form.face_membership();
    tf::intersect::search_face_pairs_self(
        form, _converter, kernel.tolerance_int(), ws,
        [&, tag](workspace_t &w, bool is_self) {
          tf::intersect::self_process(w, is_self, form, tag, mel, fm, kernel);
        });
  }

  /// One finalize for both record families, routed per record: a self
  /// record (`tag == tag_other`) duplicates through the self fan with
  /// shared-vertex sentinel delivery; a cross record through the pair
  /// duplicator.
  ///
  /// Without self records there are no sentinels, and dedup before
  /// duplication is the historical pairwise order — kept byte-stable.
  /// With self records duplication runs first: the duplicator delivers
  /// shared-vertex records (sentinel ids = point count + offset vertex
  /// id) to the pairs its fan completes, the sentinels resolve to
  /// points, and one dedup canonicalizes everything uniformly. Under
  /// SoS the sentinel base is zero — the perturbation moves a shared
  /// vertex off the pair's intersection and the SoS predicates emit the
  /// complete chords themselves, so delivery stays off.
  template <typename ApplyToForm>
  auto finalize_build(tf::local_value<workspace_t> &ws,
                      const ApplyToForm &apply_to_form, Index n_tags,
                      bool with_self, bool sos,
                      const tf::buffer<Index> &vertex_offsets) {
    tf::buffer<intersection_t> raw;
    tf::buffer<tf::exact::pt3<Int>> points;
    tf::intersect::merge_face_pair_workspaces(ws, raw, points);
    if (points.size() == 0)
      return base_t::finalize(n_tags);

    auto sentinel_base = (!with_self || sos) ? Index(0) : Index(points.size());
    auto duplicator = [&](const intersection_t &rec, auto &buffer) {
      if (rec.tag == rec.tag_other) {
        apply_to_form(int(rec.tag), [&](const auto &form) {
          tf::intersect::duplicate_intersection_self(
              form.faces(), form.face_membership(), form.manifold_edge_link(),
              rec, buffer,
              sentinel_base == Index(0)
                  ? Index(0)
                  : sentinel_base + vertex_offsets[std::size_t(rec.tag)]);
        });
      } else {
        apply_to_form(int(rec.tag), [&](const auto &form0) {
          apply_to_form(int(rec.tag_other), [&](const auto &form1) {
            tf::intersect::duplicate_intersection(
                form0.faces(), form0.face_membership(),
                form0.manifold_edge_link(), form1.faces(),
                form1.face_membership(), form1.manifold_edge_link(), rec,
                buffer);
          });
        });
      }
    };

    if (!with_self) {
      tf::intersect::dedup_coincident_points(raw, points);
      base_t::_intersection_points = std::move(points);
      tf::generic_generate(raw, base_t::_intersections, duplicator);
      return base_t::finalize(n_tags);
    }

    tf::generic_generate(raw, base_t::_intersections, duplicator);
    if (sentinel_base != Index(0))
      tf::intersect::resolve_shared_vertex_ids(base_t::_intersections, points,
                                               sentinel_base, apply_to_form,
                                               vertex_offsets, _converter);
    tf::intersect::dedup_coincident_points(base_t::_intersections, points);
    base_t::_intersection_points = std::move(points);
    base_t::finalize(n_tags);
  }

  tf::exact::vertex_converter<Int, RealType, Dims> _converter;
};

} // namespace tf
