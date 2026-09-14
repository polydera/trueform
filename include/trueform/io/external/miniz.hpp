// miniz 3.0.2 by Rich Geldreich <richgel99@gmail.com>, MIT license --
// see ./miniz/LICENSE. Vendored in ./miniz/ with two C++-required edits:
// the three `extern MINIZ_EXPORT` declarations drop their `extern`, and
// the C tentative definition of s_tdefl_num_probes becomes its one
// initialized definition. This wrapper is the one include the library
// uses.
//
// Every miniz function is given internal linkage (MINIZ_EXPORT static), so
// each translation unit carries its own copy and no symbol leaves it. The
// archive, stdio and time tiers are compiled out; what remains is the
// deflate/inflate core and the crc32 the gzip framing needs.
#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>

#define MINIZ_EXPORT static
#define MINIZ_NO_STDIO
#define MINIZ_NO_TIME
#define MINIZ_NO_ARCHIVE_APIS
#define MINIZ_NO_ARCHIVE_WRITING_APIS

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4505)
#endif

#include "./miniz/miniz.h"
#include "./miniz/miniz.c"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
