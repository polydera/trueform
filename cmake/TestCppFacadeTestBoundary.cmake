cmake_minimum_required(VERSION 3.16)

foreach(_required TRUEFORM_SOURCE_DIR TRUEFORM_BOUNDARY_CHECKER
                  TRUEFORM_BOUNDARY_TEST_BINARY_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

get_filename_component(TRUEFORM_SOURCE_DIR "${TRUEFORM_SOURCE_DIR}" REALPATH)
get_filename_component(TRUEFORM_BOUNDARY_CHECKER
  "${TRUEFORM_BOUNDARY_CHECKER}" REALPATH)
get_filename_component(_root "${TRUEFORM_BOUNDARY_TEST_BINARY_DIR}" ABSOLUTE)
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY "${_root}")

function(_run_boundary_case _name _contents _expect_success _expected_text)
  set(_case_root "${_root}/${_name}")
  file(MAKE_DIRECTORY "${_case_root}")
  set(_extension "cpp")
  if(ARGC GREATER 4)
    set(_extension "${ARGV4}")
  endif()
  file(WRITE "${_case_root}/probe.${_extension}" "${_contents}")

  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DTRUEFORM_SOURCE_DIR=${TRUEFORM_SOURCE_DIR}"
      "-DTRUEFORM_CPP_TEST_ROOT=${_case_root}"
      -P "${TRUEFORM_BOUNDARY_CHECKER}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  set(_output "${_stdout}${_stderr}")

  if(_expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "Boundary self-test '${_name}' unexpectedly failed:\n${_output}")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR
        "Boundary self-test '${_name}' unexpectedly passed")
    endif()
    if(NOT _output MATCHES "${_expected_text}")
      message(FATAL_ERROR
        "Boundary self-test '${_name}' did not report '${_expected_text}':\n"
        "${_output}")
    endif()
  endif()
endfunction()

_run_boundary_case(allowed
  "#include \"trueform/cpp/core/nd_array.hpp\"\n"
  "TRUE" "")
# Append allowed direct public-signature tokens separately because the helper's
# fixed argument shape treats each quoted argument as a new parameter.
file(APPEND "${_root}/allowed/probe.cpp"
  "tf::buffer<int> *storage = nullptr;\n"
  "auto expression = tf::csg::op(0);\n"
  "auto value = tf::cpp::make_sphere_mesh(1.0, 4, 4);\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DTRUEFORM_SOURCE_DIR=${TRUEFORM_SOURCE_DIR}"
    "-DTRUEFORM_CPP_TEST_ROOT=${_root}/allowed"
    -P "${TRUEFORM_BOUNDARY_CHECKER}"
  RESULT_VARIABLE _allowed_result
  OUTPUT_VARIABLE _allowed_stdout
  ERROR_VARIABLE _allowed_stderr
)
if(NOT _allowed_result EQUAL 0)
  message(FATAL_ERROR
    "Boundary allowed-token self-test failed:\n${_allowed_stdout}${_allowed_stderr}")
endif()

_run_boundary_case(forbidden_include
  "#include \"trueform/geometry/make_sphere_mesh.hpp\"\n"
  "FALSE" "direct core include" "cc")
_run_boundary_case(forbidden_alias
  "auto operation = tf::parallel_fill;\n"
  "FALSE" "direct core token tf::parallel_fill" "cxx")
_run_boundary_case(forbidden_wrapped
  "auto result = (tf::parallel_fill)(0, 0);\n"
  "FALSE" "direct core token tf::parallel_fill" "hxx")
_run_boundary_case(forbidden_multiline
  "auto operation = tf::\n  parallel_fill;\n"
  "FALSE" "direct core token tf::parallel_fill" "ipp")

file(REMOVE_RECURSE "${_root}")
message(STATUS "C++ facade test boundary self-test passed")
