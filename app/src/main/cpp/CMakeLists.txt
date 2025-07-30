#set(CMAKE_SYSTEM_NAME Android)
#set(CMAKE_ANDROID_NDK D:/AndroidSDK/ndk/android-ndk-r27d)
#set(CMAKE_TOOLCHAIN_FILE D:/AndroidSDK/ndk/android-ndk-r27d/build/cmake/android.toolchain.cmake)
cmake_minimum_required(VERSION 3.22.1)

project("yolov13")

set(OpenCV_DIR D:/ncnn/opencv-mobile-4.9.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR D:/ncnn/ncnn-20250503-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(${CMAKE_PROJECT_NAME} SHARED
		ndkcamera.cpp ndkcamera.h
		yolo.cpp yolo.h
		native-lib.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
		ncnn ${OpenCV_LIBS} camera2ndk mediandk
		android
		log)