/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
enum class edge_orientation : signed char {
  forward = 0,
  reverse = 1,
  bidirectional = 3
};
}
