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
#include "../sqrt.hpp"
#include <algorithm>

namespace tf::core::impl {

/// Update corner bounds with a single point's projection
template <typename T>
void update_corners(T &minx, T &maxx, T &miny, T &maxy, T px, T py, T pz, T cz,
                    T radsqr) {
  const T a = tf::sqrt(T(0.5)); // 1/sqrt(2)

  T dx, dy, u, t;

  if (px > maxx) {
    if (py > maxy) {
      // top-right corner
      dx = px - maxx;
      dy = py - maxy;
      u = dx * a + dy * a;
      t = (a * u - dx) * (a * u - dx) + (a * u - dy) * (a * u - dy) +
          (cz - pz) * (cz - pz);
      u = u - tf::sqrt(std::max<T>(radsqr - t, T(0)));
      if (u > T(0)) {
        maxx += u * a;
        maxy += u * a;
      }
    } else if (py < miny) {
      // bottom-right corner
      dx = px - maxx;
      dy = py - miny;
      u = dx * a - dy * a;
      t = (a * u - dx) * (a * u - dx) + (-a * u - dy) * (-a * u - dy) +
          (cz - pz) * (cz - pz);
      u = u - tf::sqrt(std::max<T>(radsqr - t, T(0)));
      if (u > T(0)) {
        maxx += u * a;
        miny -= u * a;
      }
    }
  } else if (px < minx) {
    if (py > maxy) {
      // top-left corner
      dx = px - minx;
      dy = py - maxy;
      u = dy * a - dx * a;
      t = (-a * u - dx) * (-a * u - dx) + (a * u - dy) * (a * u - dy) +
          (cz - pz) * (cz - pz);
      u = u - tf::sqrt(std::max<T>(radsqr - t, T(0)));
      if (u > T(0)) {
        minx -= u * a;
        maxy += u * a;
      }
    } else if (py < miny) {
      // bottom-left corner
      dx = px - minx;
      dy = py - miny;
      u = -dx * a - dy * a;
      t = (-a * u - dx) * (-a * u - dx) + (-a * u - dy) * (-a * u - dy) +
          (cz - pz) * (cz - pz);
      u = u - tf::sqrt(std::max<T>(radsqr - t, T(0)));
      if (u > T(0)) {
        minx -= u * a;
        miny -= u * a;
      }
    }
  }
}

} // namespace tf::core::impl
