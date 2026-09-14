if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR
    "double/int64 dynamic domain-label link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

foreach(expected_object IN ITEMS
    domain_labels_double_int64_3d.cpp.o
    domain_labels_result.cpp.o)
  string(FIND "${link_map}" "${expected_object}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "double/int64 dynamic domain-label link omitted required member: ${expected_object}")
  endif()
endforeach()

string(REPLACE "domain_labels_double_int64_3d.cpp.o" "" unrelated_link_map
  "${link_map}")
if(unrelated_link_map MATCHES "domain_labels\\.cpp\\.o|domain_labels_(float|double)_int(32|64)_3d\\.cpp\\.o")
  message(FATAL_ERROR
    "double/int64 dynamic domain-label link extracted a legacy or sibling operation shard")
endif()

message(STATUS
  "Double/int64 dynamic domain labels extracted their operation shard and neutral result member only")
