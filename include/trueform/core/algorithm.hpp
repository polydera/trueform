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
#include "./algorithm/assign_if.hpp"
#include "./algorithm/block_reduce.hpp"
#include "./algorithm/block_reduce_sequenced_aggregate.hpp"
#include "./algorithm/circular_decrement.hpp"
#include "./algorithm/circular_increment.hpp"
#include "./algorithm/compute_offsets.hpp"
#include "./algorithm/generate_offset_blocks.hpp"
#include "./algorithm/generic_generate.hpp"
#include "./algorithm/ids_to_index_map.hpp"
#include "./algorithm/make_equivalence_class_index_map.hpp"
#include "./algorithm/make_equivalence_class_map.hpp"
#include "./algorithm/make_unique_index_map.hpp"
#include "./algorithm/mask_to_index_map.hpp"
#include "./algorithm/mask_to_map.hpp"
#include "./algorithm/max.hpp"
#include "./algorithm/min.hpp"
#include "./algorithm/parallel_copy.hpp"
#include "./algorithm/parallel_copy_blocked.hpp"
#include "./algorithm/parallel_copy_by_map_with_nones.hpp"
#include "./algorithm/parallel_contains.hpp"
#include "./algorithm/parallel_fill.hpp"
#include "./algorithm/parallel_for.hpp"
#include "./algorithm/parallel_for_each.hpp"
#include "./algorithm/parallel_iota.hpp"
#include "./algorithm/parallel_transform.hpp"
#include "./algorithm/reduce.hpp"
#include "./algorithm/remove_by_mask_and_make_map.hpp"
#include "./algorithm/remove_if_and_make_map.hpp"
#include "./algorithm/keep_by_mask_and_make_map.hpp"
