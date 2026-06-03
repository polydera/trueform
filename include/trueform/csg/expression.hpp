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

/** @defgroup csg_expression Boolean Expression
 *  @ingroup csg
 *  Runtime boolean expression trees over operand bits, with a
 *  word-mask lowering for fast per-domain evaluation.
 */

#include "./expression/all_of.hpp"               // IWYU pragma: export
#include "./expression/any_of.hpp"               // IWYU pragma: export
#include "./expression/coalesce_words.hpp"       // IWYU pragma: export
#include "./expression/compile.hpp"              // IWYU pragma: export
#include "./expression/compile_cluster_node.hpp" // IWYU pragma: export
#include "./expression/compiled_expr.hpp"        // IWYU pragma: export
#include "./expression/complement.hpp"           // IWYU pragma: export
#include "./expression/difference.hpp"           // IWYU pragma: export
#include "./expression/expr.hpp"                 // IWYU pragma: export
#include "./expression/intersection.hpp"         // IWYU pragma: export
#include "./expression/make_children.hpp"        // IWYU pragma: export
#include "./expression/make_word_mask.hpp"       // IWYU pragma: export
#include "./expression/merge.hpp"                // IWYU pragma: export
#include "./expression/operators.hpp"            // IWYU pragma: export
#include "./expression/to_expr.hpp"              // IWYU pragma: export
