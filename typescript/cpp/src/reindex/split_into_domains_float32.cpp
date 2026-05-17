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
#include "./split_into_domains_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_reindex_split_into_domains_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<split_domains_result_t<Real>>(
      "SplitDomainsResultFloat32")
      .field("components", &split_domains_result_t<Real>::components)
      .field("labels", &split_domains_result_t<Real>::labels);

  emscripten::function("split_into_domains_float32",
                       &sync_split_into_domains<Real>);
  emscripten::function("dispatch_split_into_domains_float32",
                       &async_split_into_domains<Real>);
}
