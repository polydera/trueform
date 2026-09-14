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
#include "topology_components_impl.hpp"
#include "topology_components_instantiations.hpp"

#include <cstdint>

namespace tf::cpp {

TF_CPP_DEFINE_TOPOLOGY_COMPONENTS(std::int64_t)

} // namespace tf::cpp

#undef TF_CPP_DEFINE_TOPOLOGY_COMPONENTS
