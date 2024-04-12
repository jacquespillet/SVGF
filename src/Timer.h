#pragma once
#include <iostream>
#include <chrono>

namespace gpupt
{

class timer {
public:
    timer() : StartTime(std::chrono::high_resolution_clock::now()) {}

    ~timer() {
        Stop();
    }

    double Stop() {
        if(Started)
        {
            Started = false;
            auto EndTime = std::chrono::high_resolution_clock::now();
            auto Start = std::chrono::time_point_cast<std::chrono::microseconds>(StartTime).time_since_epoch().count();
            auto End = std::chrono::time_point_cast<std::chrono::microseconds>(EndTime).time_since_epoch().count();

            auto duration = End - Start;
            double MS = duration * 0.001;

            return MS;
        }
        return 0;
    }

    void Start()
    {
        Started=true;
        StartTime = (std::chrono::high_resolution_clock::now());
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> StartTime;

    bool Started=false;
};
}

