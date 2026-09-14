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
#include "../../core/point.hpp"
#include "../../core/transformation.hpp"
#include "./nifti_header.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tf::io::nifti {

/// A file's placement split into what the grid absorbs and what it cannot:
/// positive spacing, a local origin, and a unit-column world transformation
/// that is identity exactly when `posed` is false.
struct nifti_pose {
  tf::point<float, 3> spacing{1.f, 1.f, 1.f};
  tf::point<float, 3> origin{0.f, 0.f, 0.f};
  tf::transformation<float, 3> world = tf::make_identity_transformation<float, 3>();
  bool posed = false;
  bool reflecting = false;
};

inline auto positive_spacing(float value) -> float {
  return std::isfinite(value) && value > 0.f ? value : 1.f;
}

inline auto qform_is_finite(const char *header, bool swap) -> bool {
  bool finite = true;
  for (std::size_t i = 0; i < 3; ++i)
    finite = finite &&
             std::isfinite(field<float>(header, at_quatern + 4 * i, swap)) &&
             std::isfinite(field<float>(header, at_qoffset + 4 * i, swap));
  return finite;
}

/// The affine precedence is the format's: sform when its code is set and its
/// columns are usable, else qform, else pixdim alone.
inline auto decompose_pose(const char *header, bool swap) -> nifti_pose {
  float m[3][3];
  float t[3] = {0.f, 0.f, 0.f};
  bool oriented = false;

  if (field<std::int16_t>(header, at_sform_code, swap) > 0) {
    float srow[12];
    bool usable = true;
    for (std::size_t i = 0; i < 12; ++i) {
      srow[i] = field<float>(header, at_srow + 4 * i, swap);
      usable = usable && std::isfinite(srow[i]);
    }
    for (std::size_t c = 0; c < 3 && usable; ++c) {
      const auto norm = std::sqrt(srow[c] * srow[c] + srow[4 + c] * srow[4 + c] +
                                  srow[8 + c] * srow[8 + c]);
      usable = usable && std::isfinite(norm) && norm > 0.f;
    }
    if (usable) {
      for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
          m[r][c] = srow[4 * r + c];
      for (std::size_t r = 0; r < 3; ++r)
        t[r] = srow[4 * r + 3];
      oriented = true;
    }
  }

  nifti_pose pose;
  if (oriented) {
    for (std::size_t c = 0; c < 3; ++c) {
      const auto norm =
          std::sqrt(m[0][c] * m[0][c] + m[1][c] * m[1][c] + m[2][c] * m[2][c]);
      pose.spacing[c] = norm;
      for (std::size_t r = 0; r < 3; ++r)
        m[r][c] /= norm;
    }
  } else if (field<std::int16_t>(header, at_qform_code, swap) > 0 &&
             qform_is_finite(header, swap)) {
    const auto b = field<float>(header, at_quatern, swap);
    const auto c = field<float>(header, at_quatern + 4, swap);
    const auto d = field<float>(header, at_quatern + 8, swap);
    const auto a = std::sqrt(std::max(0.f, 1.f - b * b - c * c - d * d));
    const auto qfac =
        field<float>(header, at_pixdim, swap) < 0.f ? -1.f : 1.f;
    m[0][0] = a * a + b * b - c * c - d * d;
    m[0][1] = 2 * b * c - 2 * a * d;
    m[0][2] = (2 * b * d + 2 * a * c) * qfac;
    m[1][0] = 2 * b * c + 2 * a * d;
    m[1][1] = a * a + c * c - b * b - d * d;
    m[1][2] = (2 * c * d - 2 * a * b) * qfac;
    m[2][0] = 2 * b * d - 2 * a * c;
    m[2][1] = 2 * c * d + 2 * a * b;
    m[2][2] = (a * a + d * d - b * b - c * c) * qfac;
    for (std::size_t i = 0; i < 3; ++i) {
      pose.spacing[i] =
          positive_spacing(field<float>(header, at_pixdim + 4 * (i + 1), swap));
      t[i] = field<float>(header, at_qoffset + 4 * i, swap);
    }
    oriented = true;
  } else {
    for (std::size_t i = 0; i < 3; ++i)
      pose.spacing[i] =
          positive_spacing(field<float>(header, at_pixdim + 4 * (i + 1), swap));
    return pose;
  }

  const auto det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  pose.reflecting = det < 0.f;
  bool identity = true;
  for (std::size_t r = 0; r < 3 && identity; ++r)
    for (std::size_t c = 0; c < 3 && identity; ++c)
      identity = m[r][c] == (r == c ? 1.f : 0.f);
  if (identity) {
    pose.origin = tf::point<float, 3>{t[0], t[1], t[2]};
    return pose;
  }
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c)
      pose.world(r, c) = m[r][c];
    pose.world(r, 3) = t[r];
  }
  pose.posed = true;
  return pose;
}

/// The writer's inverse: the grid's spacing and origin composed through the
/// stated world transformation into the file's twelve srow values.
template <typename Transformation>
auto compose_srows(const tf::point<float, 3> &spacing,
                   const tf::point<float, 3> &origin,
                   const Transformation &world, float (&srow)[12]) -> void {
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c)
      srow[4 * r + c] = world(r, c) * spacing[c];
    srow[4 * r + 3] = world(r, 0) * origin[0] + world(r, 1) * origin[1] +
                      world(r, 2) * origin[2] + world(r, 3);
  }
}

} // namespace tf::io::nifti
