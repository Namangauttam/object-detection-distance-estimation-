# Build with MSYS2 UCRT64 g++:  powershell -ExecutionPolicy Bypass -File src\build.ps1
cmake_minimum_required(VERSION 3.20)
project(yolo_video_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party" CACHE PATH "Downloaded dependencies")
set(ORT_DIR "${THIRD_PARTY}/onnxruntime")

# OpenCV from MSYS2 (pacman -S mingw-w64-ucrt-x86_64-opencv)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)

add_executable(yolo_infer main.cpp distance.cpp)
target_include_directories(yolo_infer PRIVATE "${ORT_DIR}/include")
# Microsoft's ONNX Runtime GPU build exposes a plain C ABI, so g++ can link the DLL directly.
target_link_libraries(yolo_infer PRIVATE ${OpenCV_LIBS} "${ORT_DIR}/lib/onnxruntime.dll")
target_compile_options(yolo_infer PRIVATE -O2 -Wall -Wno-unknown-pragmas)
target_link_options(yolo_infer PRIVATE -static-libgcc -static-libstdc++)

# Lens calibration tool (OpenCV 5 split calib3d into calib + objdetect).
add_executable(calibrate_camera calibrate.cpp)
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
  target_link_libraries(calibrate_camera PRIVATE ${OpenCV_LIBS} opencv_imgcodecs opencv_calib opencv_objdetect opencv_geometry)
else()
  target_link_libraries(calibrate_camera PRIVATE ${OpenCV_LIBS} opencv_imgcodecs opencv_calib3d)
endif()
target_compile_options(calibrate_camera PRIVATE -O2 -Wall)
target_link_options(calibrate_camera PRIVATE -static-libgcc -static-libstdc++)

# Use the CUDA 12 / cuDNN 9 runtime DLLs bundled with PyTorch (override at runtime with CUDA_DLL_DIR).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(TORCH_LIB_DIR)
  file(TO_CMAKE_PATH "${TORCH_LIB_DIR}" TORCH_LIB_DIR)
  message(STATUS "CUDA runtime DLLs from PyTorch: ${TORCH_LIB_DIR}")
  target_compile_definitions(yolo_infer PRIVATE DEFAULT_CUDA_DLL_DIR="${TORCH_LIB_DIR}")
else()
  message(WARNING "PyTorch not found by ${Python3_EXECUTABLE}; set CUDA_DLL_DIR at runtime for GPU inference.")
endif()

# Copy ONNX Runtime DLLs next to the executable (OpenCV DLLs come from C:/msys64/ucrt64/bin on PATH).
file(GLOB ORT_DLLS "${ORT_DIR}/lib/*.dll")
add_custom_command(TARGET yolo_infer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLLS} $<TARGET_FILE_DIR:yolo_infer>)
