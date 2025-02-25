#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void computeMandelbrotCUDA(uint8_t* pixels, int width, int height,
                          float centerX, float centerY,
                          float scale, int maxIterations);

void computeMandelbrotMinimapCUDA(uint8_t* pixels, int width, int height, int maxIterations);

#ifdef __cplusplus
}
#endif
