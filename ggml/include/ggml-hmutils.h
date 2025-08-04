#ifndef GGML_HMUTILS_H
#define GGML_HMUTILS_H
#include <stdio.h>
#include <stdlib.h>

#include <string.h>  // 替代 C++ 的 <string>，用于 C 字符串操作
#include <stdbool.h> // C99 布尔类型，C 编译器需包含此头文件
#include <ctype.h>
// 仅 C++ 编译器会包含以下头文件
#ifdef __cplusplus
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
#endif

// 确保C++编译器以C语言方式处理函数
#ifdef __cplusplus
extern "C" {
#endif

// 辅助函数：将字符串转换为小写
static void toLowerCase(char *str) {
    if (str == NULL) return;
    for (int i = 0; str[i] != '\0'; i++) {
        str[i] = tolower((unsigned char)str[i]);
    }
}

// 判断文件名是否为指定后缀（不区分大小写）
static bool hasSuffix(const char *filename, const char *suffix) {
    if (filename == NULL || suffix == NULL) {
        return false;
    }

    // 查找最后一个 '.' 的位置
    const char *dotPos = strrchr(filename, '.');
    if (dotPos == NULL) {
        return false; // 没有 '.'，无后缀
    }

    // 提取后缀（从 '.' 后面开始）
    char fileSuffix[256];
    strncpy(fileSuffix, dotPos + 1, sizeof(fileSuffix) - 1);
    fileSuffix[sizeof(fileSuffix) - 1] = '\0';

    // 转换为小写进行比较
    toLowerCase(fileSuffix);

    char targetSuffix[256];
    strncpy(targetSuffix, suffix, sizeof(targetSuffix) - 1);
    targetSuffix[sizeof(targetSuffix) - 1] = '\0';
    toLowerCase(targetSuffix);

    return strcmp(fileSuffix, targetSuffix) == 0;
}

// 辅助函数：从字符串中提取子串
static void substr(const char *src, size_t start, size_t length, char *dest, size_t destSize) {
    if (src == NULL || dest == NULL) return;

    size_t srcLen = strlen(src);
    if (start >= srcLen) {
        dest[0] = '\0';
        return;
    }

    size_t actualLength = (start + length > srcLen) ? (srcLen - start) : length;
    if (actualLength >= destSize) {
        actualLength = destSize - 1;
    }

    strncpy(dest, src + start, actualLength);
    dest[actualLength] = '\0';
}

// 解析字符串为void**、void*和int
static bool parseMemoryString(const char *str,
                      void ** *add1,
                      void **add2,
                      int *add3) {
    if (str == NULL || add1 == NULL || add2 == NULL || add3 == NULL) {
        fprintf(stderr, "无效的输入参数 str:%s add1:%p add2:%p add3:%p\n",str,add1,add2,add3);
        return false;
    }

    // 1. 去除后缀 ".mem"
    const char *dotPos = strrchr(str, '.');
    if (dotPos == NULL || strcmp(dotPos, ".mem") != 0) {
        fprintf(stderr, "字符串格式错误，必须以.mem结尾\n");
        return false;
    }

    // 提取baseStr（不含.mem后缀）
    size_t baseLen = dotPos - str;
    char *baseStr = (char *)malloc(baseLen + 1);
    if (baseStr == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return false;
    }
    strncpy(baseStr, str, baseLen);
    baseStr[baseLen] = '\0';

    // 2. 按'_'分割为三部分
    char *pos1 = strchr(baseStr, '_');
    char *pos2 = strrchr(baseStr, '_');
    if (pos1 == NULL || pos2 == NULL || pos1 == pos2) {
        fprintf(stderr, "字符串格式错误，必须包含两个_\n");
        free(baseStr);
        return false;
    }

    // 提取三个部分
    char add1Str[64] = {0};
    char add2Str[64] = {0};
    char add3Str[64] = {0};

    size_t len1 = pos1 - baseStr;
    substr(baseStr, 0, len1, add1Str, sizeof(add1Str));

    size_t len2 = pos2 - pos1 - 1;
    substr(pos1 + 1, 0, len2, add2Str, sizeof(add2Str));

    substr(pos2 + 1, 0, strlen(pos2 + 1), add3Str, sizeof(add3Str));

    // 3. 转换add1 (void**)
    uint64_t add1Val;
    if (sscanf(add1Str, "%llx", &add1Val) != 1) {
        fprintf(stderr, "解析add1失败: %s\n", add1Str);
        free(baseStr);
        return false;
    }
    *add1 = (void **)add1Val;

    // 4. 转换add2 (void*)
    uint64_t add2Val;
    if (sscanf(add2Str, "%llx", &add2Val) != 1) {
        fprintf(stderr, "解析add2失败: %s\n", add2Str);
        free(baseStr);
        return false;
    }
    *add2 = (void *)add2Val;

    // 5. 转换add3 (int)
    if (sscanf(add3Str, "%d", add3) != 1) {
        fprintf(stderr, "解析add3失败: %s\n", add3Str);
        free(baseStr);
        return false;
    }

    free(baseStr);
    return true;
}
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <share.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>

// 定义映射信息结构体
typedef struct {
    HANDLE hMapFile;
    LPVOID lpMapAddress;
    size_t length;
} MapInfo;

// 配套的fclose替代函数，用于释放内存映射资源
static int fmemclose(FILE* fp) {
    if (fp == NULL) return EOF;

    // 从FILE*中获取映射信息
    void** mapInfoPtr = (void**)fp;
    MapInfo* mapInfo = (MapInfo*)*mapInfoPtr;

    // 释放内存映射资源
    UnmapViewOfFile(mapInfo->lpMapAddress);
    CloseHandle(mapInfo->hMapFile);
    free(mapInfo);

    // 关闭文件描述符
    int fd = _fileno(fp);
    int ret = fclose(fp);
    _close(fd);

    return ret;
}

static FILE *fmemopen(void *buf, size_t len, const char *type)
{
	int fd;
	FILE *fp;
	char tp[MAX_PATH - 13];
	char fn[MAX_PATH + 1];
	int * pfd = &fd;
	int retner = -1;
	char tfname[] = "MemTF_";
	if (!GetTempPathA(sizeof(tp), tp))
		return NULL;
	if (!GetTempFileNameA(tp, tfname, 0, fn))
		return NULL;
	retner = _sopen_s(pfd, fn, _O_CREAT | _O_SHORT_LIVED | _O_TEMPORARY | _O_RDWR | _O_BINARY | _O_NOINHERIT, _SH_DENYRW, _S_IREAD | _S_IWRITE);
	if (retner != 0)
		return NULL;
	if (fd == -1)
		return NULL;
	fp = _fdopen(fd, "wb+");
	if (!fp) {
		_close(fd);
		return NULL;
	}
	/*File descriptors passed into _fdopen are owned by the returned FILE * stream.If _fdopen is successful, do not call _close on the file descriptor.Calling fclose on the returned FILE * also closes the file descriptor.*/
	//GGML_LOG_INFO("++beofre fwrite:size:%lld real len\r\n",len);
    fwrite(buf, len, 1, fp);
    //fwrite(buf, 5*1024*1024*1024, 1, fp);
    //GGML_LOG_INFO("++after fwrite:size:%lld real len\r\n",len);

    //char magic_string[5]={0};
    //memcpy(magic_string,buf,4);
    //GGML_LOG_INFO("++magic_string in fmemopen is:%s and return\r\n",magic_string);
	rewind(fp);
	return fp;
}
#endif


#ifdef __cplusplus
}
#endif

// --------------------------
// 以下是 C++ 专属函数（仅 C++ 编译器可见）
// --------------------------
#ifdef __cplusplus
// 用于比较const char*的自定义比较器
struct CStringCompare {
    bool operator()(const char* a, const char* b) const {
        // 使用strcmp进行字符串比较，确保map能正确排序和查找
        return std::strcmp(a, b) < 0;
    }
};

// 判断文件名是否以.mem后缀结尾（不区分大小写）
inline bool endsWithMem(const std::string& filename) {
    // 查找最后一个 '.' 的位置
    size_t dotPos = filename.find_last_of('.');

    // 如果没有 '.' 或者 '.' 是最后一个字符，则不是.mem结尾
    if (dotPos == std::string::npos || dotPos == filename.size() - 1) {
        return false;
    }

    // 提取 '.' 后面的子字符串
    std::string suffix = filename.substr(dotPos + 1);

    // 转换为小写后比较（忽略大小写，如.MEM也视为有效）
    for (char& c : suffix) {
        c = std::tolower(c);
    }

    return suffix == "mem";
}

// 从FILE*指定偏移量读取数据到vector
inline bool readFromFile(FILE* file, size_t offset, size_t size, std::vector<char>& buffer) {
    if (file == nullptr) {
        std::cerr << "错误：文件指针为空" << std::endl;
        return false;
    }

    // 移动文件指针到指定偏移量
    if (fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
        std::cerr << "错误：无法移动到偏移量 " << offset << std::endl;
        return false;
    }

    // 调整vector大小以容纳数据
    // buffer.resize(size);

    // 读取数据
    size_t bytesRead = fread(buffer.data(), 1, size, file);
    rewind(file);
    if (bytesRead != size) {
        std::cerr << "警告：实际读取 " << bytesRead << " 字节，预期 " << size << " 字节" << std::endl;
        // 可以选择截断vector以匹配实际读取的大小
        buffer.resize(bytesRead);
        return false;
    }
    return true;
}

#endif // __cplusplus
#endif // GGML_HMUTILS_H
