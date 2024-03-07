set(SIMULATOR_SRCS
    types.hpp sim.hpp sim.inl sim.cpp
)

add_library(madwarp_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(madwarp_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(madwarp_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(madwarp_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        madwarp_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madwarp_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madwarp_mgr PRIVATE
        madwarp_gpu_srcs
        madrona_mw_gpu
        madrona_cuda
    )
endif ()

target_compile_definitions(madwarp_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_warp_proto_sim
    bindings.cpp
)

target_link_libraries(madrona_warp_proto_sim PRIVATE
    madwarp_mgr
)

if (TARGET madrona_viz)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer PRIVATE
        madrona_mw_core madwarp_mgr madrona_viz)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )

    if (TARGET madrona_cuda)
        target_link_libraries(viewer PRIVATE
            madrona_cuda
        )
    endif()
endif()
