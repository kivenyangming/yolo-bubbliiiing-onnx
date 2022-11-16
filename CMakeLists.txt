# 声明要求的 cmake 最低版本
cmake_minimum_required( VERSION 3.1 )

# 声明一个 cmake 工程
project( pro )

# 寻找opencv的路径
#set(OpenCV_DIR E:/OPENCV/build) #WIN

#添加OPENCV库 指定OpenCV版本，代码如下OpenCV4.5.5 REQUIRED & 如果不需要指定OpenCV版本，代码如下
find_package(OpenCV REQUIRED)

#添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

#显示OpenCV_INCLUDE_DIRS的值
message(${OpenCV_INCLUDE_DIRS})

# 添加头文件
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 添加连接库
link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# 指定生成目标
add_executable(main main.cpp lib/yolo.cpp include/yolo.h)
# 将库文件链接到可执行程序上
target_link_libraries( main  ${OpenCV_LIBS})

# cd build
# cmake  ..
# cmake --build . # 或make
