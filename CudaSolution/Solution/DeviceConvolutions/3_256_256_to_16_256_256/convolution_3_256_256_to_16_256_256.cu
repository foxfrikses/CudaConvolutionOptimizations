#include "convolution_3_256_256_to_16_256_256.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>

#include <stdexcept>

#include "CudaApi.h"

namespace convolution_3_256_256_to_16_256_256 {

  struct Conv_3_by_3_Params {
    const float* input;  // [nInputChannel][shape.x][shape.y]
    const float* filter; // [nInputChannel][3][3]
    const float* bias;   // [nInputChannel]
    float* output;       // [nInputChannel][shape.x][shape.y]
  };

  struct Conv_1_by_1_Params {
    const float* input;  // [nInputChannel][shape.x][shape.y]
    const float* filter; // [nOutputChannel][nInputChannel][1][1]
    const float* bias;   // [nOutputChannel]
    float* output;       // [nOutputChannel][shape.x][shape.y]
  };


#define input_id(c_, x_, y_)  ((x_) + 256 * (y_) + 256 * 256 * (c_))
#define output_id(c_, x_, y_) ((x_) + 256 * (y_) + 256 * 256 * (c_))
#define filter_id(c_, x_, y_) ((x_) + 3 * (y_) + 3 * 3 * (c_))

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

  __global__ void Conv_3_by_3(
    Conv_3_by_3_Params params
  )
  {
    if (threadIdx.x > 255 || blockIdx.x > 3) {
      return;
    }

    const int outputX = threadIdx.x;
    const int inputX = threadIdx.x + 1;
    const int channel = blockIdx.x;

    __shared__ float input[3][258];

    if (threadIdx.x == 0) {
      input[0][0] = 0.0;
      input[1][0] = 0.0;
      input[2][0] = 0.0;
      input[0][257] = 0.0;
      input[1][257] = 0.0;
      input[2][257] = 0.0;
    }

    input[0][inputX] = 0.0;
    input[1][inputX] = params.input[input_id(channel, outputX, 0)];

    const float bias = params.bias[channel];

    float filter[3][3];
    memcpy(filter, params.filter + 3 * 3 * channel, 3 * 3 * sizeof(float));

    int y;
    for (y = 0; y < 255; ++y) {
      input[(y + 2) % 3][inputX] =
        params.input[input_id(channel, outputX, y + 1)];

      __syncthreads();

      params.output[output_id(channel, outputX, y)] =
        Multiply(&input[(y + 0) % 3][outputX],
                 &input[(y + 1) % 3][outputX],
                 &input[(y + 2) % 3][outputX],
                 filter) + bias;

      __syncthreads();
    }

    input[(y + 2) % 3][inputX] = 0.0f;

    __syncthreads();

    params.output[output_id(channel, outputX, y)] =
      Multiply(&input[(y + 0) % 3][outputX],
               &input[(y + 1) % 3][outputX],
               &input[(y + 2) % 3][outputX],
               filter) + bias;
  }

  __global__ void Conv_1_by_1(
    Conv_1_by_1_Params params
  )
  {
    float filter[3];
    memcpy(filter, params.filter + blockIdx.x * 3, 3 * sizeof(float));

    const float bias = params.bias[blockIdx.x];

    for (int y = 0; y < 256; ++y) {
      params.output[output_id(blockIdx.x, threadIdx.x, y)] =
        params.input[input_id(0, threadIdx.x, y)] * filter[0] +
        params.input[input_id(1, threadIdx.x, y)] * filter[1] +
        params.input[input_id(2, threadIdx.x, y)] * filter[2] +
        bias;
    }
  }
}

void Convolution_3_256_256_to_16_256_256::Compute(DeviceConvolutionParams& params) const
{
  if (params.nInputChannel != 3 ||
      params.nOutputChannel != 16 ||
      params.shape.x != 256 ||
      params.shape.y != 256) {
    throw std::invalid_argument("Error Convolution_3_256_256_to_16_256_256. Invalid argument");
  }

  using namespace convolution_3_256_256_to_16_256_256;
  {
    Conv_3_by_3_Params params_3_by_3;
    params_3_by_3.input = params.input.get();
    params_3_by_3.filter = params.dwFilter.get();
    params_3_by_3.bias = params.dwBias.get();
    params_3_by_3.output = params.interimOutput.get();

    Conv_3_by_3<<<3, 256>>>(params_3_by_3);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }

  {
    Conv_1_by_1_Params params_1_by_1;
    params_1_by_1.input = params.interimOutput.get();
    params_1_by_1.filter = params.dFilter.get();
    params_1_by_1.bias = params.dBias.get();
    params_1_by_1.output = params.output.get();

    Conv_1_by_1<<<16, 256>>>(params_1_by_1);
    Cuda::CheckForLastError();
    Cuda::DeviceSynchronize();
  }
}
