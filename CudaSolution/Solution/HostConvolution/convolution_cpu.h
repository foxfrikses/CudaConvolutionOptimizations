#pragma once

#include <vector>
#include <vector_types.h>

namespace cpu {
  struct Params {
    const float* input;         // [nInputChannel][shape.x][shape.y]
    const float* dwFilter;      // [nInputChannel][3][3]
    const float* dwBias;        // [nInputChannel]
    const float* dFilter;       // [nOutputChannel][nInputChannel][1][1]
    const float* dBias;         // [nOutputChannel]
          float* interimOutput; // [nInputChannel][shape.x][shape.y]
          float* output;        // [nOutputChannel][shape.x][shape.y]

    int nInputChannel;
    int nOutputChannel;
    int2 shape;
  };

  void Convolution(const Params& params);
}
