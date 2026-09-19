cmake_minimum_required(VERSION 3.16)
project(video_upscaler_stream_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(video-upscaler-engine main.cpp)
if (MSVC)
  target_compile_options(video-upscaler-engine PRIVATE /W4)
else()
  target_compile_options(video-upscaler-engine PRIVATE -Wall -Wextra -Wpedantic)
endif()
