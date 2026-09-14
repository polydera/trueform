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

#include "./exact_dot.hpp"
#include "./exact_lane.hpp"
#include "./intercept_ladder.hpp"
#include "./pool_records.hpp"
#include "./tally_census.hpp"
#include "./wide_magnitude.hpp"
#include "./wide_to_double.hpp"

#include <cmath>

namespace tf::exact::door::pool {

/// Whether two members stand further apart along the cell's line than
/// `band`.
///
/// The exact statement compares squared physical distances without ever
/// forming the irrational one:
///
///     C^2 H  >  band^2 B_i^2 B_j^2,   C = A_j B_i - A_i B_j,   H = L . L.
///
/// THREE ROUTES AND ONE ANSWER, chosen by WIDTH. The screen is the double
/// heights and their own error bound, and it decides every pair that does
/// not stand within its slack of the band. Two members whose aligned normals
/// are the LINE ITSELF reduce to the offsets alone, `(D_j - D_i)^2 >
/// band^2 H`, which is what a wall family is made of. Everything else takes
/// the general form on the product rung where a width proof admits it, and
/// in limb scratch where it does not — the general expression reaches 596
/// bits on the int32 lattice.
///
/// The width the proof is stated against is the LANE'S, so the same
/// statement routes at either lattice and neither rung is named here.
template <typename Int>
auto ladder_gap_exceeds(const ladder_point<Int> &a, const ladder_point<Int> &b,
                        const typename exact_lane<Int>::product_type
                            &square_length,
                        const typename exact_lane<Int>::product_type &band,
                        election_census &census) -> bool {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;
  using magnitude_type = wide_magnitude<exact_lane<Int>::product_bits>;
  const unsigned rung = unsigned(exact_lane<Int>::product_bits);

  tally_census(census.gap_tests);
  const double reach = wide_to_double(band);
  const double apart = std::fabs(a.height - b.height);
  const double doubt = a.slack + b.slack;
  if (apart - doubt > reach) {
    tally_census(census.gap_filtered);
    return true;
  }
  if (apart + doubt < reach) {
    tally_census(census.gap_filtered);
    return false;
  }

  const product_type &span = band;
  if (a.reach == b.reach) {
    const coefficient_type step = b.residual - a.residual;
    const product_type left(exact_product<Int>(product_type(step),
                                              product_type(step)));
    if (a.reach == square_length) {
      tally_census(census.gap_exact);
      return left > exact_product<Int>(exact_product<Int>(span, span),
                                       square_length);
    }
    const unsigned bits = wide_bit_width(left) + wide_bit_width(square_length);
    const unsigned against =
        2 * wide_bit_width(span) + 2 * wide_bit_width(a.reach);
    if (bits <= rung && against <= rung) {
      tally_census(census.gap_exact);
      return exact_product<Int>(left, square_length) >
             exact_product<Int>(exact_product<Int>(span, span),
                                exact_product<Int>(a.reach, a.reach));
    }
    tally_census(census.gap_limbs);
    const auto scaled = wide_magnitude_of<magnitude_type>(span * a.reach);
    return wide_compare(
               wide_multiply(wide_magnitude_of<magnitude_type>(left),
                             wide_magnitude_of<magnitude_type>(square_length)),
               wide_multiply(scaled, scaled)) > 0;
  }

  const product_type across =
      exact_product<Int>(product_type(b.residual), a.reach) -
      exact_product<Int>(product_type(a.residual), b.reach);
  const unsigned bits = 2 * wide_bit_width(across) +
                        wide_bit_width(square_length);
  const unsigned against = 2 * wide_bit_width(span) +
                           2 * wide_bit_width(a.reach) +
                           2 * wide_bit_width(b.reach);
  if (bits <= rung && against <= rung) {
    tally_census(census.gap_exact);
    const product_type scaled = exact_product<Int>(
        exact_product<Int>(span, a.reach), b.reach);
    return exact_product<Int>(exact_product<Int>(across, across),
                              square_length) >
           exact_product<Int>(scaled, scaled);
  }
  tally_census(census.gap_limbs);
  const auto left = wide_magnitude_of<magnitude_type>(across);
  const auto scaled =
      wide_multiply(wide_multiply(wide_magnitude_of<magnitude_type>(span),
                                  wide_magnitude_of<magnitude_type>(a.reach)),
                    wide_magnitude_of<magnitude_type>(b.reach));
  return wide_compare(
             wide_multiply(wide_multiply(left, left),
                           wide_magnitude_of<magnitude_type>(square_length)),
             wide_multiply(scaled, scaled)) > 0;
}

} // namespace tf::exact::door::pool
