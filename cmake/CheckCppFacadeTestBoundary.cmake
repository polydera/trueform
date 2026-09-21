cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED TRUEFORM_SOURCE_DIR)
  message(FATAL_ERROR "TRUEFORM_SOURCE_DIR is required")
endif()
get_filename_component(TRUEFORM_SOURCE_DIR "${TRUEFORM_SOURCE_DIR}" REALPATH)

if(DEFINED TRUEFORM_CPP_TEST_ROOT)
  get_filename_component(_test_root "${TRUEFORM_CPP_TEST_ROOT}" REALPATH)
else()
  set(_test_root "${TRUEFORM_SOURCE_DIR}/cpp/tests")
endif()
if(NOT IS_DIRECTORY "${_test_root}")
  message(FATAL_ERROR "C++ facade test directory does not exist: ${_test_root}")
endif()

# A caller holds its own geometry and assembles views over it, so the headers
# that DECLARE the carriers and views the token list already allows are the
# caller's too. Everything else in core stays shut: these are storage,
# placements and the factories that make views of them, never an algorithm.
set(_allowed_core_includes
  "trueform/core/buffer.hpp"
  "trueform/core/curves_buffer.hpp"
  "trueform/core/faces.hpp"
  "trueform/core/points.hpp"
  "trueform/core/points_buffer.hpp"
  "trueform/core/polygons_buffer.hpp"
  "trueform/core/range.hpp"
  "trueform/core/segments_buffer.hpp"
  "trueform/core/static_size.hpp"
  "trueform/core/transformation.hpp"
  "trueform/core/transformation_view.hpp"
  "trueform/core/unit_vectors.hpp"
  "trueform/core/unit_vectors_buffer.hpp"
)

# What ONE file is allowed that the layer's callers are not: a frozen ABI probe
# names the configuration headers of the release it froze.
set(_allowed_direct_includes
  "cpp/tests/installed_consumer/legacy_remesh_abi.cpp|trueform/remesh/decimate_config.hpp"
  "cpp/tests/installed_consumer/legacy_remesh_abi.cpp|trueform/remesh/isotropic_remesh_config.hpp"
  "cpp/tests/installed_consumer/legacy_remesh_abi.cpp|trueform/remesh/simplify_config.hpp"
)

# Public tf::cpp signatures intentionally expose these core storage,
# configuration, angle, transformation, and expression value types. Scoped
# enum values under an allowlisted token are allowed as well.
set(_allowed_direct_tokens
  "tf::arrangement_config"
  "tf::boolean_config"
  "tf::boolean_op"
  "tf::buffer"
  "tf::csg::expr"
  "tf::csg::inside"
  "tf::csg::op"
  "tf::csg::selection"
  "tf::csg::selection_t"
  "tf::curves_buffer"
  "tf::decimate_config"
  "tf::deg"
  "tf::domain_config"
  "tf::dynamic_size"
  "tf::has_frame_policy"
  "tf::identity_transformation"
  "tf::intersect_config"
  "tf::intersect_mode"
  "tf::isotropic_remesh_config"
  "tf::make_edges"
  "tf::make_faces"
  "tf::make_points"
  "tf::make_range"
  "tf::make_transformation_view"
  "tf::make_unit_vectors"
  "tf::points_buffer"
  "tf::polygons_buffer"
  "tf::rad"
  "tf::segments_buffer"
  "tf::simplify_config"
  "tf::small_vector"
  "tf::transformation"
  "tf::transformation_view"
  "tf::triangulation_type"
  "tf::unit_vectors_buffer"
)

file(GLOB_RECURSE _test_sources LIST_DIRECTORIES FALSE
  "${_test_root}/*.cpp"
  "${_test_root}/*.cc"
  "${_test_root}/*.cxx"
  "${_test_root}/*.h"
  "${_test_root}/*.hpp"
  "${_test_root}/*.hxx"
  "${_test_root}/*.ipp"
  "${_test_root}/*.inl"
)

set(_violations)
foreach(_source IN LISTS _test_sources)
  file(RELATIVE_PATH _relative "${TRUEFORM_SOURCE_DIR}" "${_source}")
  file(READ "${_source}" _contents)

  string(REGEX MATCHALL
    "#[ \t]*include[ \t]*[<\"]trueform/[^>\"\r\n]+[>\"]"
    _includes "${_contents}")
  foreach(_directive IN LISTS _includes)
    string(REGEX REPLACE
      ".*[<\"](trueform/[^>\"]+)[>\"].*" "\\1" _include "${_directive}")
    # The layer's front door is trueform/cpp.hpp and everything under
    # trueform/cpp/; both are the tf::cpp surface this check exists to require.
    if(NOT _include MATCHES "^trueform/cpp/" AND
       NOT _include STREQUAL "trueform/cpp.hpp")
      set(_include_key "${_relative}|${_include}")
      if(NOT _include IN_LIST _allowed_core_includes AND
         NOT _include_key IN_LIST _allowed_direct_includes)
        list(APPEND _violations
          "${_relative}: direct core include <${_include}>")
      endif()
    endif()
  endforeach()

  # Scan qualified tokens, not merely direct calls. Whitespace is accepted
  # around scope operators so aliases, address-taking, wrapped calls, and
  # multiline spellings cannot bypass the check.
  string(REGEX MATCHALL
    "tf[ \t\r\n]*::[ \t\r\n]*[A-Za-z_][A-Za-z0-9_]*([ \t\r\n]*::[ \t\r\n]*[A-Za-z_][A-Za-z0-9_]*)*"
    _tokens "${_contents}")
  foreach(_raw_token IN LISTS _tokens)
    string(REGEX REPLACE "[ \t\r\n]" "" _token "${_raw_token}")
    if(_token STREQUAL "tf::cpp" OR _token MATCHES "^tf::cpp::")
      continue()
    endif()

    set(_allowed FALSE)
    foreach(_prefix IN LISTS _allowed_direct_tokens)
      if(_token STREQUAL _prefix)
        set(_allowed TRUE)
        break()
      endif()
      string(FIND "${_token}" "${_prefix}::" _prefix_position)
      if(_prefix_position EQUAL 0)
        set(_allowed TRUE)
        break()
      endif()
    endforeach()
    if(NOT _allowed)
      list(APPEND _violations "${_relative}: direct core token ${_token}")
    endif()
  endforeach()
endforeach()

if(_violations)
  list(REMOVE_DUPLICATES _violations)
  list(JOIN _violations "\n  " _formatted)
  message(FATAL_ERROR
    "C++ facade tests must exercise public tf::cpp APIs. Found:\n  ${_formatted}\n"
    "Add a tf::cpp capability instead of referencing a header-only core "
    "algorithm. Core data/configuration inputs require a narrow, reviewed "
    "allowlist entry."
  )
endif()

list(LENGTH _test_sources _source_count)
message(STATUS "C++ facade test boundary check passed (${_source_count} files)")
