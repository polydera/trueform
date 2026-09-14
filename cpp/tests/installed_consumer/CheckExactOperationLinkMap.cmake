if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT DEFINED SHARD_LABEL OR SHARD_LABEL STREQUAL "")
  message(FATAL_ERROR "SHARD_LABEL is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "${SHARD_LABEL} link map does not exist: ${LINK_MAP}")
endif()
if(NOT DEFINED EXPECTED_OBJECTS_CSV OR EXPECTED_OBJECTS_CSV STREQUAL "")
  message(FATAL_ERROR "EXPECTED_OBJECTS_CSV is required for ${SHARD_LABEL}")
endif()
if(NOT DEFINED FORBIDDEN_PATTERN OR FORBIDDEN_PATTERN STREQUAL "")
  message(FATAL_ERROR "FORBIDDEN_PATTERN is required for ${SHARD_LABEL}")
endif()

file(READ "${LINK_MAP}" link_map)
set(unrelated_link_map "${link_map}")
string(REPLACE "," ";" expected_objects "${EXPECTED_OBJECTS_CSV}")
foreach(expected_object IN LISTS expected_objects)
  string(FIND "${link_map}" "${expected_object}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "${SHARD_LABEL} link omitted required archive member: ${expected_object}")
  endif()
  string(REPLACE "${expected_object}" "" unrelated_link_map
    "${unrelated_link_map}")
endforeach()

if(unrelated_link_map MATCHES "${FORBIDDEN_PATTERN}")
  message(FATAL_ERROR
    "${SHARD_LABEL} link extracted a forbidden generalized, dynamic, or sibling archive member")
endif()

message(STATUS
  "${SHARD_LABEL} extracted its exact allowed archive closure")
