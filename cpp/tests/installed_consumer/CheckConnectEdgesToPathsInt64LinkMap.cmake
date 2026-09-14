if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "int64 edge-path link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

string(FIND "${link_map}" "topology_components_int64.cpp.o" own_position)
if(own_position EQUAL -1)
  message(FATAL_ERROR
    "int64 edge-path link omitted its own production shard")
endif()

string(FIND "${link_map}" "topology_components_int32.cpp.o" other_position)
if(NOT other_position EQUAL -1)
  message(FATAL_ERROR
    "int64 edge-path link extracted the int32 production shard")
endif()

message(STATUS
  "int64 edge paths extracted only their own topology component shard")
