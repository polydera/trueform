# Reindex binding sources
# Add new files here when creating new bindings
set(MODULE_REINDEX_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/reindex_by_ids.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reindex_by_mask.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_components.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_int3float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_int3double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_int643float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_int643double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_intdynfloat3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_intdyndouble3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_int64dynfloat3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/split_into_domains_int64dyndouble3d.cpp
)
