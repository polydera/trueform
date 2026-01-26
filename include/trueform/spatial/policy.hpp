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

/// @file
/// @ingroup spatial_policies
/// @brief Policy tag helpers for spatial tree structures.
///
/// Provides functions to attach spatial tree metadata to range types using
/// a policy-based composition pattern. Use `tf::tag()` or specific `tf::tag_tree()`
/// and `tf::tag_mod_tree()` functions to attach tree data.

#pragma once

#include "./policy/tree.hpp"
