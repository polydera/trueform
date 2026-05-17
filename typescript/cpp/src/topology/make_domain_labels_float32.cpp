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
#include "./make_domain_labels_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_topology_make_domain_labels_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<domain_labels_result_t<Real>>(
      "DomainLabelsResultFloat32")
      .field("labels", &domain_labels_result_t<Real>::labels)
      .field("nDomains", &domain_labels_result_t<Real>::n_domains)
      .field("outerShellLabel",
             &domain_labels_result_t<Real>::outer_shell_label);

  emscripten::function("make_domain_labels_float32",
                       &sync_make_domain_labels<Real>);
  emscripten::function("dispatch_make_domain_labels_float32",
                       &async_make_domain_labels<Real>);
}
