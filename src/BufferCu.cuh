#pragma once
#include <stdint.h>

namespace gpupt
{

class bufferCu {
public:
    bufferCu() = default;
    bufferCu(int Size, void *InitData = nullptr);
    ~bufferCu();
    void Reallocate(int Size, void *InitData = nullptr);
    void Destroy();
    void updateData(const void* data, size_t dataSize);
    void updateData(size_t offset, const void* data, size_t dataSize);

    void *Data;
    uint32_t Size;
};
}
