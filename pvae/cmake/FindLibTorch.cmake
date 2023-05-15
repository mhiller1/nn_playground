
set(LIBTORCH_ROOT "${LIBTORCH_ROOT}" CACHE PATH "LibTorch root directory")

find_library(LIBTORCH_C10_LIBRARY 	NAMES c10 		PATHS "${LIBTORCH_ROOT}" PATH_SUFFIXES lib)
find_library(LIBTORCH_TORCH_LIBRARY 	NAMES torch 		PATHS "${LIBTORCH_ROOT}" PATH_SUFFIXES lib)
find_library(LIBTORCH_TORCH_CPU_LIBRARY NAMES torch_cpu 	PATHS "${LIBTORCH_ROOT}" PATH_SUFFIXES lib)
find_path(LIBTORCH_INCLUDE_DIR 		NAMES torch/library.h   PATHS "${LIBTORCH_ROOT}" PATH_SUFFIXES include)

mark_as_advanced(LIBTORCH_C10_LIBRARY LIBTORCH_TORCH_LIBRARY LIBTORCH_TORCH_CPU_LIBRARY LIBTORCH_TORCH_GLOBAL_DEPS_LIBRARY LIBTORCH_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibTorch DEFAULT_MSG LIBTORCH_C10_LIBRARY LIBTORCH_TORCH_LIBRARY LIBTORCH_TORCH_CPU_LIBRARY LIBTORCH_INCLUDE_DIR)

if(LIBTORCH_FOUND)
    add_library(LibTorch INTERFACE)
    target_include_directories(LibTorch INTERFACE ${LIBTORCH_INCLUDE_DIR} ${LIBTORCH_INCLUDE_DIR}/torch/csrc/api/include)
    target_link_libraries(LibTorch INTERFACE ${LIBTORCH_C10_LIBRARY} ${LIBTORCH_TORCH_LIBRARY} ${LIBTORCH_TORCH_CPU_LIBRARY})

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(LibTorch INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

