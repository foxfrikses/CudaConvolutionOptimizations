#include "original_fused_convolution.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaApi.h"

// channel-first
#define id(a, c, x, y, A, C, X, Y) ((x) + (X) * ((y) + (Y) * ((c) + (C) * (a))))

#define max_(a, b) (a < b ? b : a)
#define min_(a, b) (a < b ? a : b)
#define max(a, b) (max_((a), (b)))
#define min(a, b) (min_((a), (b)))

#define ReLU(v) (max((v), 0.0f))

namespace original_fused_convolution {
  struct Conv_Params {
    const float* input;         // [nInputChannel][shape.x][shape.y]
    const float* dwFilter;      // [nInputChannel][3][3]
    const float* dwBias;        // [nInputChannel]
    const float* dFilter;       // [nOutputChannel][nInputChannel][1][1]
    const float* dBias;         // [nOutputChannel]
    float*       output;        // [nOutputChannel][shape.x][shape.y]

    int nInputChannel;
    int nOutputChannel;

    int2 shape;
  };

  extern __shared__ float buffer[/* nInputChannel * tileSize.x * tileSize.y * sizeof(float) */];

  __global__ void Conv(Conv_Params p) {
    const int2 tileCount{(p.shape.x + gridDim.x - 1) / gridDim.x,
                         (p.shape.y + gridDim.y - 1) / gridDim.y};

    const int2 tileSize = {gridDim.x, gridDim.y};

    const int x_begin = blockIdx.x * gridDim.x;
    const int y_begin = blockIdx.y * gridDim.y;

    const int x_end = min(p.shape.x, x_begin + gridDim.x);
    const int y_end = min(p.shape.y, y_begin + gridDim.y);


    float t;
    const int inputChannel = threadIdx.x;
    if (inputChannel < p.nInputChannel) {
      for (int y = y_begin; y < y_end; ++y) {
        const int fyBegin = (y == 0 ? 1 : 0);
        const int fyEnd = (y == p.shape.y - 1 ? 2 : 3);

        for (int x = x_begin; x < x_end; ++x) {
          const int fxBegin = (x == 0 ? 1 : 0);
          const int fxEnd = (x == p.shape.x - 1 ? 2 : 3);

          t = 0.0f;

          for (int fy = fyBegin; fy < fyEnd; fy++) {
            for (int fx = fxBegin; fx < fxEnd; fx++) {
              t +=
                p.input[id(0,
                           inputChannel,
                           x + fx - 1,
                           y + fy - 1,
                           1,
                           p.nInputChannel,
                           p.shape.x,
                           p.shape.y)] *
                p.dwFilter[id(0,
                              inputChannel,
                              fx,
                              fy,
                              1,
                              p.nInputChannel,
                              3,
                              3)];
            }
          }

          buffer[id(0,
                    inputChannel,
                    x - x_begin,
                    y - y_begin,
                    1,
                    p.nInputChannel,
                    tileSize.x,
                    tileSize.y)] = t + p.dwBias[inputChannel];
        }
      }
    }

    __syncthreads();

    const int outputChannel = threadIdx.x;
    if (outputChannel < p.nOutputChannel) {
      for (int y = y_begin; y < y_end; ++y)
        for (int x = x_begin; x < x_end; ++x) {
          t = 0.0f;

          for (int inputChannel = 0; inputChannel < p.nInputChannel; ++inputChannel) {
            t +=
              buffer[id(0,
                        inputChannel,
                        x - x_begin,
                        y - y_begin,
                        1,
                        p.nInputChannel,
                        tileSize.x,
                        tileSize.y)] *
              p.dFilter[id(outputChannel,
                           inputChannel,
                           0,
                           0,
                           p.nOutputChannel,
                           p.nInputChannel,
                           1,
                           1)];
          }

          p.output[id(0,
                      outputChannel, 
                      x, 
                      y, 
                      1,
                      p.nOutputChannel, 
                      p.shape.x, 
                      p.shape.y)] = t + p.dBias[outputChannel];
        }
    }
  }

}

OriginalFusedConvolution::OriginalFusedConvolution(int2 tileCount)
  : tileCount(tileCount)
{}

void OriginalFusedConvolution::Compute(DeviceConvolutionParams& params) const
{
  using namespace original_fused_convolution;

  {
    Conv_Params p;
    p.input          = params.input.get();
    p.dwFilter       = params.dwFilter.get();
    p.dwBias         = params.dwBias.get();
    p.dFilter        = params.dFilter.get();
    p.dBias          = params.dBias.get();
    p.output         = params.output.get();
    p.nInputChannel  = params.nInputChannel;
    p.nOutputChannel = params.nOutputChannel;
    p.shape          = params.shape;

    dim3 dimGrid((params.shape.x + tileCount.x - 1) / tileCount.x,
                 (params.shape.y + tileCount.y - 1) / tileCount.y);
    dim3 dimBlock(max(params.nInputChannel, params.nOutputChannel));
    int sharedMemoryAmount = params.nInputChannel * 
      dimGrid.x * dimGrid.y * sizeof(float);

    Conv<<<dimGrid, dimBlock, sharedMemoryAmount>>>(p);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }
}
