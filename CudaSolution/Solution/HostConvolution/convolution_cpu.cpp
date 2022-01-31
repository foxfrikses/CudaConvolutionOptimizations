#include "convolution_cpu.h"

#include <iostream>

namespace {
  int id(int2 coord, int2 shape, 
         int channel, int channelCount, 
         int batch = 0) {
    int res = batch;
    res = res * channelCount + channel;
    res = res * shape.y + coord.y;
    res = res * shape.x + coord.x;
    return res;
  }
}

namespace cpu {
  int2 GetFilterBeginEnd(int x, int shapeX) {
    return {(x > 0 ? 0 : 1),
            (x < shapeX - 1 ? 3 : 2)};
  }

  void Convolution(const Params& params)
  {
    for (int channel = 0; channel < params.nInputChannel; ++channel) {
      for (int y = 0; y < params.shape.y; ++y) {
        const auto [fyBegin, fyEnd] = GetFilterBeginEnd(y, params.shape.y);

        for (int x = 0; x < params.shape.x; ++x) {
          const auto [fxBegin, fxEnd] = GetFilterBeginEnd(x, params.shape.x);

          float t = 0.0f;
          for (int fx = fxBegin; fx < fxEnd; ++fx) {
            for (int fy = fyBegin; fy < fyEnd; ++fy) {
              t +=
                params.input[id({x - 1 + fx, y - 1 + fy},
                                params.shape,
                                channel, 
                                params.nInputChannel)] *
                params.dwFilter[id({fx, fy},
                                   {3, 3},
                                   channel,
                                   params.nInputChannel)];

            }
          }

          params.interimOutput[id({x, y},
                                  params.shape,
                                  channel,
                                  params.nInputChannel)] = t + params.dwBias[channel];
        }
      }
    }
    
    for (int outputChannel = 0; outputChannel < params.nOutputChannel; ++outputChannel) {
      for (int x = 0; x < params.shape.x; ++x) {
        for (int y = 0; y < params.shape.y; ++y) {

          float t = 0.0f;

          for (int channel = 0; channel < params.nInputChannel; ++channel) {
            t +=
              params.interimOutput[id({x, y},
                                      params.shape,
                                      channel,
                                      params.nInputChannel)] *
              params.dFilter[id({0, 0},
                                {1, 1},
                                channel,
                                params.nInputChannel,
                                outputChannel)];
          }

          params.output[id({x, y},
                           params.shape,
                           outputChannel,
                           params.nOutputChannel)] = t + params.dBias[outputChannel];
        }
      }
    }
  }
}
