#include "size_flexible_optimized_convolution.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>

#include "CudaApi.h"

namespace size_flexible_optimezed_convolution {
  struct Conv_3_by_3_Params {
    const float* input;  // [nInputChannel][shape.x][shape.y]
    const float* filter; // [nInputChannel][3][3]
    const float* bias;   // [nInputChannel]
    float* output;       // [nInputChannel][shape.x][shape.y]

    int nChannel;
    int2 shape;
  };

  struct Conv_1_by_1_Params {
    const float* input;  // [nInputChannel][shape.x][shape.y]
    const float* filter; // [nOutputChannel][nInputChannel][1][1]
    const float* bias;   // [nOutputChannel]
    float* output;       // [nOutputChannel][shape.x][shape.y]

    int nInputChannel;
    int nOutputChannel;
    int2 shape;
  };

#define input_id(c, x, y, X, Y)  ((x) + (X) * ((y) + (Y) * (c)))
#define output_id(c, x, y, X, Y) ((x) + (X) * ((y) + (Y) * (c)))
#define filter_3_3_id(c, x, y)   ((x) + 3 * (y) + 3 * 3 * (c))
#define filter_1_1_id(a, c, C)   ((c) + (C) * (a))

  __device__ float Multiply(float* first,
                            float* second,
                            float* third,
                            float(&filter)[3][3])
  {
    return
      first[0] * filter[0][0] +
      first[1] * filter[0][1] +
      first[2] * filter[0][2] +
      second[0] * filter[1][0] +
      second[1] * filter[1][1] +
      second[2] * filter[1][2] +
      third[0] * filter[2][0] +
      third[1] * filter[2][1] +
      third[2] * filter[2][2];
  }

  extern __shared__ float input_3_by_3[/* 3 * (2 + shape.x) */];
  #define input_3_by_3_id(y, x, X) ((y) * (X + 2) + x)
  /*
    3 * (2 + shape.x) * sizeof(float) of shared memory
    channelCount of blocks
    shape.x of threads
  */
  __global__ void Conv_3_by_3(
    Conv_3_by_3_Params params
  )
  {
    if (threadIdx.x >= params.shape.x || blockIdx.x > params.nChannel) {
      return;
    }

    const int outputX = threadIdx.x;
    const int inputX = threadIdx.x + 1;
    const int channel = blockIdx.x;

    if (threadIdx.x == 0) {
      input_3_by_3[input_3_by_3_id(0, 0, params.shape.x)] = 0.0;
      input_3_by_3[input_3_by_3_id(1, 0, params.shape.x)] = 0.0;
      input_3_by_3[input_3_by_3_id(2, 0, params.shape.x)] = 0.0;
      input_3_by_3[input_3_by_3_id(0, params.shape.x + 1, params.shape.x)] = 0.0;
      input_3_by_3[input_3_by_3_id(1, params.shape.x + 1, params.shape.x)] = 0.0;
      input_3_by_3[input_3_by_3_id(2, params.shape.x + 1, params.shape.x)] = 0.0;
    }

    input_3_by_3[input_3_by_3_id(0, inputX, params.shape.x)] = 0.0;
    input_3_by_3[input_3_by_3_id(1, inputX, params.shape.x)] =
      params.input[input_id(channel, outputX, 0, params.shape.x, params.shape.y)];

    const float bias = params.bias[channel];

    float filter[3][3];
    memcpy(filter, params.filter + 3 * 3 * channel, 3 * 3 * sizeof(float));

    int y;
    for (y = 0; y < params.shape.y - 1; ++y) {
      input_3_by_3[input_3_by_3_id((y + 2) % 3, inputX, params.shape.x)] =
        params.input[input_id(channel, outputX, y + 1, params.shape.x, params.shape.y)];

      __syncthreads();

      params.output[output_id(channel, outputX, y, params.shape.x, params.shape.y)] =
        Multiply(&input_3_by_3[input_3_by_3_id((y + 0) % 3, outputX, params.shape.x)],
                 &input_3_by_3[input_3_by_3_id((y + 1) % 3, outputX, params.shape.x)],
                 &input_3_by_3[input_3_by_3_id((y + 2) % 3, outputX, params.shape.x)],
                 filter) + bias;

      __syncthreads();
    }

    input_3_by_3[input_3_by_3_id((y + 2) % 3, inputX, params.shape.x)] = 0.0f;

    __syncthreads();

    params.output[output_id(channel, outputX, y, params.shape.x, params.shape.y)] =
      Multiply(&input_3_by_3[input_3_by_3_id((y + 0) % 3, outputX, params.shape.x)],
               &input_3_by_3[input_3_by_3_id((y + 1) % 3, outputX, params.shape.x)],
               &input_3_by_3[input_3_by_3_id((y + 2) % 3, outputX, params.shape.x)],
               filter) + bias;
  }

  extern __shared__ float filter[/* nInputChannel */];
  /*
    nInputChannel * sizeof(float) of shared memory
    nOutputChannel of blocks
    shape.x of threads
  */
  __global__ void Conv_1_by_1(
    Conv_1_by_1_Params params
  )
  {
    const int channel = blockIdx.x;

    for (int i = threadIdx.x; i < params.nInputChannel; i += blockDim.x) {
      filter[i] = params.filter[channel * params.nInputChannel + i];
    }
    __syncthreads();

    const float bias = params.bias[channel];

    float t;

    const int x = threadIdx.x;
    for (int y = 0; y < params.shape.y; ++y) {
      t = 0.0f;

      for (int inputChannel = 0; inputChannel < params.nInputChannel; ++inputChannel) {
        t += params.input[input_id(inputChannel, x, y, params.shape.x, params.shape.y)] *
          filter[inputChannel];
      }

      params.output[output_id(channel, x, y, params.shape.x, params.shape.y)] = t + bias;
    }
  }
}

void SizeFlexibleOptimizedConvolution::Compute(DeviceConvolutionParams& params) const
{
  using namespace size_flexible_optimezed_convolution;
  {
    Conv_3_by_3_Params params_3_by_3;
    params_3_by_3.input = params.input.get();
    params_3_by_3.filter = params.dwFilter.get();
    params_3_by_3.bias = params.dwBias.get();
    params_3_by_3.output = params.interimOutput.get();

    params_3_by_3.shape = params.shape;
    params_3_by_3.nChannel = params.nInputChannel;

    const int shared_mem_amount = 3 * (2 + params.shape.x) * sizeof(float);
    const int block_count = params.nInputChannel;
    const int thread_count = params.shape.x;
    Conv_3_by_3<<<block_count, thread_count, shared_mem_amount>>>(params_3_by_3);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }

  {
    Conv_1_by_1_Params params_1_by_1;
    params_1_by_1.input = params.interimOutput.get();
    params_1_by_1.filter = params.dFilter.get();
    params_1_by_1.bias = params.dBias.get();
    params_1_by_1.output = params.output.get();

    params_1_by_1.shape = params.shape;
    params_1_by_1.nInputChannel = params.nInputChannel;
    params_1_by_1.nOutputChannel = params.nOutputChannel;

    const int shared_mem_amount = params.nInputChannel * sizeof(float);
    const int block_count = params.nOutputChannel;
    const int thread_count = params.shape.x;

    Conv_1_by_1<<<block_count, thread_count, shared_mem_amount>>>(params_1_by_1);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }
}
