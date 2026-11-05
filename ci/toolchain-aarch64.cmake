# toolchain-aarch64.cmake

# 设置目标系统名称
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64) # 或 arm64

# 指定 C 和 C++ 编译器
# 确保这些路径与你安装的交叉编译工具链路径相符
SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# 指定目标架构的查找路径
# 这是非常重要的一部分，告诉 CMake 在哪里找到目标系统的库和头文件
# 你可能需要根据你的具体工具链安装路径进行调整
# SET(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu") # 通常是交叉工具链的根目录

# 设置 CMake 的查找模式
# CMAKE_FIND_ROOT_PATH_MODE_PROGRAM：当查找程序时，只在 CMAKE_FIND_ROOT_PATH 中查找
# CMAKE_FIND_ROOT_PATH_MODE_LIBRARY：当查找库时，只在 CMAKE_FIND_ROOT_PATH 中查找
# CMAKE_FIND_ROOT_PATH_MODE_INCLUDE：当查找头文件时，只在 CMAKE_FIND_ROOT_PATH 中查找
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# 如果你的目标系统是基于 Debian/Ubuntu 的，可能还需要这些
# SET(CMAKE_SYSROOT /usr/aarch64-linux-gnu) # 如果你的工具链有独立的 sysroot 目录

# (可选) 定义一些交叉编译相关的 flags
# 例如，如果你需要为特定的 ARM 架构优化
# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+crc -mcpu=cortex-a72")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crc -mcpu=cortex-a72")