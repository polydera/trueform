if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT DEFINED SHARD_LABEL OR SHARD_LABEL STREQUAL "")
  message(FATAL_ERROR "SHARD_LABEL is required")
endif()
if(NOT DEFINED EXPECTED_OBJECT OR EXPECTED_OBJECT STREQUAL "")
  message(FATAL_ERROR "EXPECTED_OBJECT is required")
endif()
if(NOT DEFINED FORBIDDEN_PATTERN OR FORBIDDEN_PATTERN STREQUAL "")
  message(FATAL_ERROR "FORBIDDEN_PATTERN is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "${SHARD_LABEL} link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

string(FIND "${link_map}" "${EXPECTED_OBJECT}" expected_position)
if(expected_position EQUAL -1)
  message(FATAL_ERROR
    "${SHARD_LABEL} link omitted its requested shard: ${EXPECTED_OBJECT}")
endif()

# Remove every occurrence of the expected member before applying the family-wide
# rejection pattern. This catches the legacy member and every sibling axis shard
# without allowing a requested member to satisfy its own negative check.
string(REPLACE "${EXPECTED_OBJECT}" "" unrelated_link_map "${link_map}")
if(unrelated_link_map MATCHES "${FORBIDDEN_PATTERN}")
  message(FATAL_ERROR
    "${SHARD_LABEL} link extracted a legacy or sibling operation shard matching: ${FORBIDDEN_PATTERN}")
endif()

message(STATUS
  "${SHARD_LABEL} link extracted only its requested operation shard")
