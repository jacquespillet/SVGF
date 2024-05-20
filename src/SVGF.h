#pragma once
#include <stdint.h>
#include <memory>

namespace gpupt
{
    struct svgfBuffers
    {
        std::shared_ptr<buffer> ColourBuffer;
        std::shared_ptr<buffer> VarianceBuffer;
        std::shared_ptr<buffer> MomentsBuffer;
        std::shared_ptr<buffer> HistoryBufferColour;
        std::shared_ptr<buffer> HistoryBufferMoments;
        std::shared_ptr<buffer> MotionVectors;
        void Init(uint32_t Width, uint32_t Height);
    };

    

}