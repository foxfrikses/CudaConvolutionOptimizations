#include "original_convolution.h"

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

namespace original_convolution {

  struct Conv_3_by_3_Params {
    thrust::device_ptr<const float> input;   // [channelCount][shape.x][shape.y]
    thrust::device_ptr<const float> filter;  // [channelCount][3][3]
    thrust::device_ptr<const float> bias;    // [channelCount][1][1]
    thrust::device_ptr<float> output;        // [channelCount][shape.x][shape.y]

    int2 shape;
    int channelCount;
  };

  struct Conv_1_by_1_Params {
    thrust::device_ptr<const float> input;  // [nInputChannel][shape.x][shape.y]
    thrust::device_ptr<const float> filter; // [nOutputChannel][nInputChannel][1][1]
    thrust::device_ptr<const float> bias;   // [nOutputChannel]
    thrust::device_ptr<float> output;       // [nOutputChannel][shape.x][shape.y]

    int2 shape;
    int nInputChannel;
    int nOutputChannel;
  };

  __global__ void Conv_3_by_3(
    Conv_3_by_3_Params params
  )
  {
    const int channel = threadIdx.x;

    const int x_begin = blockIdx.x * gridDim.x;
    const int x_end = min(x_begin + gridDim.x, params.shape.x);

    const int y_begin = blockIdx.y * gridDim.y;
    const int y_end = min(y_begin + gridDim.y, params.shape.y);

    float t;

    for (int y = y_begin; y < y_end; ++y) {
      const int fyBegin = (y == 0 ? 1 : 0);
      const int fyEnd = (y == params.shape.y - 1 ? 2 : 3);

      for (int x = x_begin; x < x_end; ++x) {
        const int fxBegin = (x == 0 ? 1 : 0);
        const int fxEnd = (x == params.shape.x - 1 ? 2 : 3);

        t = 0;

        for (int fy = fyBegin; fy < fyEnd; ++fy) {
          for (int fx = fxBegin; fx < fxEnd; ++fx) {
            t +=
              params.input[id(0,
                              channel,
                              x + fx - 1,
                              y + fy - 1,
                              1,
                              params.channelCount,
                              params.shape.x,
                              params.shape.y)] *
              params.filter[id(0,
                               channel,
                               fx,
                               fy,
                               1,
                               params.channelCount,
                               3,
                               3)];
          }
        }

        t += params.bias[channel];

        params.output[id(0,
                         channel,
                         x,
                         y,
                         1,
                         params.channelCount,
                         params.shape.x,
                         params.shape.y)] = t;
      }
    }
  }

  __global__ void Conv_1_by_1(
    Conv_1_by_1_Params params
  )
  {
    const int outputChannel = threadIdx.x;

    const int output_y_begin = blockIdx.y * gridDim.y;
    const int output_y_end = min((blockIdx.y + 1) * gridDim.y, params.shape.y);

    const int output_x_begin = blockIdx.x * gridDim.x;
    const int output_x_end = min((blockIdx.x + 1) * gridDim.x, params.shape.x);

    float t;
    for (int output_y = output_y_begin; output_y < output_y_end; ++output_y) {
      for (int output_x = output_x_begin; output_x < output_x_end; ++output_x) {
        t = 0;

        for (int inputChannel = 0;
             inputChannel < params.nInputChannel;
             ++inputChannel) {
          t +=
            params.input[id(0,
                            inputChannel,
                            output_x,
                            output_y,
                            1,
                            params.nInputChannel,
                            params.shape.x,
                            params.shape.y)] *
            params.filter[id(outputChannel,
                             inputChannel,
                             0,
                             0,
                             params.nOutputChannel,
                             params.nInputChannel,
                             1,
                             1)];
        }

        t += params.bias[outputChannel];

        params.output[id(0,
                         outputChannel,
                         output_x,
                         output_y,
                         1,
                         params.nOutputChannel,
                         params.shape.x,
                         params.shape.y)] = t;
      }
    }
  }
}

OriginalConvolution::OriginalConvolution(int2 tileCount)
  : tileCount(tileCount)
{}

void OriginalConvolution::Compute(DeviceConvolutionParams& params) const
{
  using namespace original_convolution;

  {
    Conv_3_by_3_Params params_3_by_3;
    params_3_by_3.input = params.input;
    params_3_by_3.filter = params.dwFilter;
    params_3_by_3.bias = params.dwBias;
    params_3_by_3.output = params.interimOutput;
    params_3_by_3.shape = params.shape;
    params_3_by_3.channelCount = params.nInputChannel;

    dim3 dimGrid((params_3_by_3.shape.x + tileCount.x - 1) / tileCount.x,
                 (params_3_by_3.shape.y + tileCount.y - 1) / tileCount.y);
    dim3 dimBlock(params_3_by_3.channelCount);

    Conv_3_by_3<<<dimGrid, dimBlock>>>(params_3_by_3);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }

  {
    Conv_1_by_1_Params params_1_by_1;
    params_1_by_1.input = params.interimOutput;
    params_1_by_1.filter = params.dFilter;
    params_1_by_1.bias = params.dBias;
    params_1_by_1.output = params.output;
    params_1_by_1.shape = params.shape;
    params_1_by_1.nInputChannel = params.nInputChannel;
    params_1_by_1.nOutputChannel = params.nOutputChannel;

    dim3 dimGrid((params_1_by_1.shape.x + tileCount.x - 1) / tileCount.x,
                 (params_1_by_1.shape.y + tileCount.y - 1) / tileCount.y);
    dim3 dimBlock(params_1_by_1.nOutputChannel);

    Conv_1_by_1<<<dimGrid, dimBlock>>>(params_1_by_1);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }
}
