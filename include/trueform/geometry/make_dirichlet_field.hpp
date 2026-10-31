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
#include "../core/algorithm/mask_to_index_map.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/resolved_output_real.hpp"
#include "../core/static_size.hpp"
#include "../core/views/indirect_range.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/policy/face_membership.hpp"
#include "./dirichlet_field_result.hpp"
#include "./dirichlet_field_status.hpp"
#include "./dirichlet_order.hpp"
#include "./laplacian/assemble_laplacian_system.hpp"
#include "./laplacian/laplacian_form.hpp"
#include "./laplacian/laplacian_is_anchored.hpp"
#include "./laplacian/laplacian_system.hpp"
#include "./laplacian/make_laplacian_form_pattern.hpp"
#include "./laplacian/solve_dirichlet_form.hpp"
#include "./laplacian/state_dirichlet_load.hpp"
#include "./laplacian/state_laplacian_form_values.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup geometry
/// @brief The field that holds `fixed_values` at `fixed_ids` and vanishes
/// under the @ref tf::dirichlet_order's operator at every other point.
///
/// `L` is the Laplacian the corner weights state — `corner_weights[3 f + c]`
/// is the weight of the edge of triangle `f` opposite its corner `c`, summed
/// over the triangles an edge borders — and `M` the barycentric lumped mass.
/// `tf::none` states the cotangent weights. The free rows are solved by sparse
/// Cholesky, refined once and accepted against their true residual; see
/// @ref tf::geometry::laplacian_system for the sign convention.
///
/// The system is definite where every weight is positive and every connected
/// run of free points reaches a fixed one, which is proven before the factor
/// runs; the result's @ref tf::dirichlet_field_status names what refused.
///
/// The ids must be distinct and in range. The face membership is read where
/// tagged and built otherwise. The solve runs in double precision and the
/// field is stated in the resolved output type once, at the end.
///
/// @tparam OutputCoordinateType The type the field is stated in; `tf::none`
/// states the mesh's own.
/// @param polygons A triangle mesh.
/// @param corner_weights `3 * F` weights, or `tf::none` for the cotangent.
/// @param fixed_ids The points the field is held at.
/// @param fixed_values The value at each of them.
/// @param order The operator the field vanishes under.
/// @return The @ref tf::dirichlet_field_result.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Weights, typename Range0, typename Range1>
auto make_dirichlet_field(const tf::polygons<Policy> &polygons,
                          const Weights &corner_weights,
                          const Range0 &fixed_ids, const Range1 &fixed_values,
                          tf::dirichlet_order order)
    -> tf::dirichlet_field_result<tf::resolved_output_real_t<
        OutputCoordinateType, tf::coordinate_type<Policy>>> {
  static_assert(tf::static_size_v<decltype(polygons.faces()[0])> == 3,
                "a Dirichlet field is stated over triangles");
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "a Dirichlet field is stated in a floating-point type");
  if constexpr (!tf::has_face_membership_policy<Policy>) {
    tf::face_membership<Index> membership;
    membership.build(polygons);
    return tf::make_dirichlet_field<OutputCoordinateType>(
        polygons | tf::tag(membership), corner_weights, fixed_ids, fixed_values,
        order);
  } else {
    const std::size_t n = polygons.points().size();
    tf::buffer<double> held;
    held.allocate(n);
    tf::parallel_fill(held, 0.0);
    tf::buffer<char> free_rows;
    free_rows.allocate(n);
    tf::parallel_fill(free_rows, char(1));
    tf::parallel_for(
        std::size_t(0), std::size_t(fixed_ids.size()),
        [&](std::size_t first, std::size_t last) {
          for (std::size_t k = first; k < last; ++k) {
            held[std::size_t(fixed_ids[k])] = double(fixed_values[k]);
            free_rows[std::size_t(fixed_ids[k])] = char(0);
          }
        },
        tf::checked);
    const auto answer = [&held](tf::dirichlet_field_status status) {
      if constexpr (std::is_same_v<RealOut, double>)
        return tf::dirichlet_field_result<RealOut>{std::move(held), status};
      else {
        tf::dirichlet_field_result<RealOut> out{{}, status};
        out.field.allocate(held.size());
        tf::parallel_transform(
            held, out.field, [](double value) { return RealOut(value); },
            tf::checked);
        return out;
      }
    };

    tf::geometry::laplacian_assembly_scratch assembly;
    tf::geometry::laplacian_system<Index> system;
    bool stated;
    if constexpr (std::is_same_v<Weights, tf::none_t>)
      stated = tf::geometry::assemble_laplacian_system(
          polygons, tf::geometry::cotangent_weights{}, assembly, system);
    else
      stated = tf::geometry::assemble_laplacian_system(
          polygons,
          [&corner_weights](std::size_t t, std::size_t slot,
                            const tf::point<double, 3>(&)[3], double) {
            return double(corner_weights[3 * t + (slot + 2) % 3]);
          },
          assembly, system);
    if (!stated)
      return answer(tf::dirichlet_field_status::refused_operator);

    const auto rows = tf::mask_to_index_map<Index>(free_rows);
    tf::buffer<Index> stack;
    tf::buffer<char> seen;
    if (!tf::geometry::laplacian_is_anchored(system, rows.kept_ids(), rows.f(),
                                             stack, seen))
      return answer(tf::dirichlet_field_status::refused_unanchored);
    const bool bends = order == tf::dirichlet_order::biharmonic;
    tf::geometry::laplacian_form<Index> form;
    tf::geometry::make_laplacian_form_pattern(system, rows.kept_ids(), rows.f(),
                                              bends, form);
    tf::geometry::state_laplacian_form_values(system, rows.kept_ids(), rows.f(),
                                              bends ? 0.0 : 1.0,
                                              bends ? 1.0 : 0.0, 0.0, form);
    tf::buffer<double> load;
    tf::geometry::state_dirichlet_load(form, held, 1, load);
    tf::geometry::dirichlet_solve_scratch<Index> scratch;
    tf::buffer<double> solved;
    const auto status =
        tf::geometry::solve_dirichlet_form(form, load, 1, scratch, solved);
    if (status == tf::dirichlet_field_status::solved)
      tf::parallel_copy(solved, tf::make_indirect_range(rows.kept_ids(), held));
    return answer(status);
  }
}

} // namespace tf
