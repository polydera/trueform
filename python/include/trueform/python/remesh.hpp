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

#include <nanobind/nanobind.h>

namespace tf::py {

void register_decimated(nanobind::module_ &m);

void register_isotropic_remeshed(nanobind::module_ &m);

void register_simplified(nanobind::module_ &m);

void register_remesh(nanobind::module_ &m);

} // namespace tf::py
