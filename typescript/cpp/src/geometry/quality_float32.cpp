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

#include "./quality_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_quality_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<face_quality_result_t<Real>>(
      "FaceQualityResultFloat32")
      .field("quality", &face_quality_result_t<Real>::quality)
      .field("minAngle", &face_quality_result_t<Real>::min_angle)
      .field("maxAngle", &face_quality_result_t<Real>::max_angle)
      .field("aspectRatio", &face_quality_result_t<Real>::aspect_ratio);

  emscripten::value_object<dihedral_angles_result_t<Real>>(
      "DihedralAnglesResultFloat32")
      .field("edges", &dihedral_angles_result_t<Real>::edges)
      .field("angles", &dihedral_angles_result_t<Real>::angles);

  emscripten::function("face_quality_float32", &sync_face_quality<Real>);
  emscripten::function("dispatch_face_quality_float32",
                       &async_face_quality<Real>);
  emscripten::function("dihedral_angles_float32", &sync_dihedral_angles<Real>);
  emscripten::function("dispatch_dihedral_angles_float32",
                       &async_dihedral_angles<Real>);
}
