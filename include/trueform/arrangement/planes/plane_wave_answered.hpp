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

#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "./make_plane_refinement_cut_mask.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// THE MASK, PA-side: the source faces this arrangement has ANSWERED FOR —
/// the world's own, because promoting one would state a single
/// `(tag, object)` twice and the exposure reads that as one slot, plus every
/// face an entrance has already promoted or declined, which is what makes
/// the entrance terminate.
///
/// It is DENSE per tag, over the source form's own face space, because the
/// world's descriptors do not ascend past its base extent — an entrance
/// appends what it promoted — so no order of theirs can be searched. The
/// seed is @ref tf::arrangement::make_plane_refinement_cut_mask, the one
/// producer of "which source faces does this world hold", and a build that
/// never reaches an entrance never pays for it.
///
/// `count` is the wave's own progress term: an entrance answers for at
/// least one face it had not answered for, so the count strictly grows
/// every time one fires.
template <typename Index> struct plane_wave_answered {
  tf::core::std_vector<tf::buffer<char>> mask;
  std::size_t count = 0;

  auto seeded() const -> bool { return mask.size() != 0; }

  template <typename World, typename ApplyToForm>
  auto seed(const World &world, Index n_tags, const ApplyToForm &apply_to_form)
      -> void {
    mask = make_plane_refinement_cut_mask(world, n_tags, apply_to_form);
    for (const auto &tag_mask : mask)
      count += std::size_t(
          std::count(tag_mask.begin(), tag_mask.end(), char(1)));
  }

  auto answered(Index tag, Index object) const -> bool {
    return mask[std::size_t(tag)][std::size_t(object)] != char(0);
  }

  /// Answer for a face named in the flat source space the entrance states
  /// its entrants in.
  template <typename FaceOffsets>
  auto answer(const FaceOffsets &face_offsets, Index flat) -> void {
    const auto tag = std::size_t(
        std::upper_bound(face_offsets.begin(), face_offsets.end(), flat) -
        face_offsets.begin() - 1);
    auto &bit = mask[tag][std::size_t(flat - face_offsets[tag])];
    count += std::size_t(bit == char(0));
    bit = char(1);
  }
};

} // namespace tf::arrangement
