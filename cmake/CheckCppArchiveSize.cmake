cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED TRUEFORM_CPP_ARCHIVE OR TRUEFORM_CPP_ARCHIVE STREQUAL "")
  message(FATAL_ERROR "TRUEFORM_CPP_ARCHIVE is required")
endif()
if(NOT EXISTS "${TRUEFORM_CPP_ARCHIVE}")
  message(FATAL_ERROR "C++ facade archive does not exist: ${TRUEFORM_CPP_ARCHIVE}")
endif()
if(NOT DEFINED TRUEFORM_CPP_ARCHIVE_MAX_BYTES OR
   TRUEFORM_CPP_ARCHIVE_MAX_BYTES STREQUAL "")
  message(FATAL_ERROR "TRUEFORM_CPP_ARCHIVE_MAX_BYTES is required")
endif()
if(NOT TRUEFORM_CPP_ARCHIVE_MAX_BYTES MATCHES "^[0-9]+$")
  message(FATAL_ERROR
    "TRUEFORM_CPP_ARCHIVE_MAX_BYTES must be a positive integer")
endif()

file(SIZE "${TRUEFORM_CPP_ARCHIVE}" _archive_size)
if(_archive_size GREATER TRUEFORM_CPP_ARCHIVE_MAX_BYTES)
  math(EXPR _overage "${_archive_size} - ${TRUEFORM_CPP_ARCHIVE_MAX_BYTES}")
  message(FATAL_ERROR
    "C++ facade archive is ${_archive_size} bytes, exceeding the "
    "${TRUEFORM_CPP_ARCHIVE_MAX_BYTES}-byte safety limit by ${_overage} bytes. "
    "Reduce template instantiation duplication or split an operation lane; do "
    "not raise the limit toward the 4 GiB archive boundary.")
endif()

math(EXPR _headroom "${TRUEFORM_CPP_ARCHIVE_MAX_BYTES} - ${_archive_size}")
message(STATUS
  "C++ facade archive size ${_archive_size} bytes "
  "(${_headroom} bytes below safety limit)")
