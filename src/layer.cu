/* Last Updated: 24.08.27. 18:30 */
#include "layer.h"
#include "cuda_profiler_api.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 
void Linear_old(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[m * N + n] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k];
      }
      out->buf[m * N + n] += b->buf[n];
    }
  }
}
*/
//cudaProfilerStart();
 //cudaError_t err;

    // Start the CUDA profiler
__global__ void Linear_kernel(float *in, float *w, float *b, float *out, size_t M, size_t N, size_t K) {
  
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
 
 if (m < M && n < N) {
        float value = 0.0f;
        for (size_t k = 0; k < K; k++) {
            value += in[m * K + k] * w[n * K + k];
        } 
        out[m * N + n] = value + b[n];
    }
}
//



void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1];

    // CUDA 스트림 생성
    //cudaStream_t stream_download, stream_execute, stream_upload1,stream_upload2,stream_upload3;
    //cudaEvent_t upload_event,calc_event;
    //스트림 생성 
    // CHECK_CUDA(cudaStreamCreate(&stream_download));
    // CHECK_CUDA(cudaStreamCreate(&stream_execute));
    // CHECK_CUDA(cudaStreamCreate(&stream_upload1));
    // //이벤트 생성
    // CHECK_CUDA(cudaEventCreate(&upload_event));
    // CHECK_CUDA(cudaEventCreate(&calc_event));
    




    // 메모리 설정
     float *d_buf, *d_w, *d_b, *d_out;
    // CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * M * K));
    // CHECK_CUDA(cudaMalloc(&d_w, sizeof(float) * K * N));
    // CHECK_CUDA(cudaMalloc(&d_b, sizeof(float) * N));
    // CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * M * N));

    // // 데이터를 GPU로 비동기 복사 (stream_download 사용)
    // //CUDA error (src/layer.cu:82): cudaErrorInvalidResourceHandle:invalid resource handle 떠서 스트림 3개 만들었는데 안됨
    
    // CHECK_CUDA(cudaMemcpyAsync(d_in, in->buf, sizeof(float) * M * K, cudaMemcpyHostToDevice, stream_upload1));
    // CHECK_CUDA(cudaMemcpyAsync(d_w, w->buf, sizeof(float) * K * N, cudaMemcpyHostToDevice, stream_upload1));
    // CHECK_CUDA(cudaMemcpyAsync(d_b, b->buf, sizeof(float) * N, cudaMemcpyHostToDevice, stream_upload1));
    // //CHECK_CUDA(cudaEventRecord(upload_event,stream_upload3));//업로드 이벤트 기록
    // //CHECK_CUDA(cudaEventRecord(upload_event,stream_upload2));
    // CHECK_CUDA(cudaEventRecord(upload_event,stream_upload1));

    // 메모리 전송이 완료된 후 커널 실행 (stream_execute 사용)
    dim3 blockDim(32, 32);
    dim3 gridDim(32,32);
     //dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    //CHECK_CUDA(cudaStreamWaitEvent(stream_execute,upload_event));  // 다운로드 완료 대기
    Linear_kernel<<<gridDim, blockDim>>>(in->d_buf, d_w, d_b, d_out, M, N, K);
    //Linear_kernel<<<gridDim, blockDim, 0, stream_execute>>>(d_buf, d_w, d_b, d_out, M, N, K);
    // // 커널 실행이 완료된 후 결과를 GPU에서 CPU로 비동기 복사 (stream_upload 사용)
    // CHECK_CUDA(cudaEventRecord(calc_event,stream_execute)); 




    // CHECK_CUDA(cudaStreamWaitEvent(stream_download,calc_event)); // 커널 실행 완료 대기
    // CHECK_CUDA(cudaMemcpyAsync(out->buf, d_out, sizeof(float) * M * N, cudaMemcpyDeviceToHost, stream_download));

    // // 업로드 완료 대기 및 스트림 동기화
    // //CHECK_CUDA(cudaStreamSynchronize(stream_upload));

    // // 스트림 해제 및 메모리 해제
    // CHECK_CUDA(cudaStreamDestroy(stream_download));
    // CHECK_CUDA(cudaStreamDestroy(stream_execute));
    // CHECK_CUDA(cudaStreamDestroy(stream_upload1));
    // //CHECK_CUDA(cudaStreamDestroy(stream_upload2));
    // //CHECK_CUDA(cudaStreamDestroy(stream_upload3));

    // CHECK_CUDA(cudaFree(d_in));
    // CHECK_CUDA(cudaFree(d_w));
    // CHECK_CUDA(cudaFree(d_b));
    // CHECK_CUDA(cudaFree(d_out));
    
}


//cudaProfilerStop();



/* Reshape 
 * @param [in]   in: [N, D]
 * @param [out] out: [N, C, H, W]
 * 'N' is the number of input tensors.
 * 'D' is the dimension of the input tensor.
 * 'C' is the number of channels.
 * 'H' is the height of the output tensor.
 * 'W' is the width of the output tensor.
 
void Reshape_old(Tensor *in, Tensor *out) {
  size_t N = in->shape[0];
  size_t D = in->shape[1];
  size_t C = out->shape[1];
  size_t H = out->shape[2];
  size_t W = out->shape[3];

  for (size_t n = 0; n < N; n++) { // 가장 작은 0부터 입력 텐서의 개수 N까지 반복
    for (size_t c = 0; c < C; c++) { // 입력 채널 0부터 C까지 반복
      for (size_t h = 0; h < H; h++) { // 0부터 height = H까지 반복
        for (size_t w = 0; w < W; w++) { // 0부터 width = W까지 반복
          out->buf[n * C * H * W + c * H * W + h * W + w] = // 인풋버퍼 
              in->buf[n * D + c * H * W + h * W + w];
        }
      }
    }
  }
}
*/

// CUDA 커널 함수 정의
static __global__ void Reshape_Kernel(float* in_buf, float* out_buf, size_t N, size_t D, size_t C, size_t H, size_t W) {
    // 3D 그리드 및 블록 구성에 대한 인덱스 계산
    size_t n = blockIdx.x;
    size_t c = blockIdx.y;
    size_t h = threadIdx.x / W;
    size_t w = threadIdx.x % W;

    // 인덱스가 올바른 범위에 있는지 확인
    if (h < H && w < W) {
        out_buf[n * C * H * W + c * H * W + h * W + w] =
            in_buf[n * D + c * H * W + h * W + w];
    }
}

// 호스트에서 호출할 Reshape 함수
void Reshape(Tensor* in, Tensor* out) {
    size_t N = in->shape[0];
    size_t D = in->shape[1];
    size_t C = out->shape[1];
    size_t H = out->shape[2];
    size_t W = out->shape[3];

    
    // GPU 메모리에 입력 및 출력 텐서 버퍼 할당
    float* d_in_buf;
    float* d_out_buf;
    // CHECK_CUDA(cudaMalloc(&d_in_buf, N * D * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&d_out_buf, N * C * H * W * sizeof(float)));

    // // 입력 버퍼를 GPU로 복사
    // CHECK_CUDA(cudaMemcpy(d_in_buf, in->buf, N * D * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(N, C, 1);
    dim3 blockDim(H * W);

    // CUDA 커널 호출
    Reshape_Kernel<<<gridDim, blockDim>>>(d_in_buf, d_out_buf, N, D, C, H, W);
    // CHECK_CUDA(cudaDeviceSynchronize());

    // // 결과를 GPU에서 호스트로 복사
    // CHECK_CUDA(cudaMemcpy(out->buf, d_out_buf, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost););

    // // GPU 메모리 해제
    // CHECK_CUDA(cudaFree(d_in_buf));
    // CHECK_CUDA(cudaFree(d_out_buf));
}



/* ConvTranspose2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C, K, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *    
 *    OH = (H - 1) * stride - 2 * pad + dilation * (R - 1) + output_pad + 1
 *    OW = (W - 1) * stride - 2 * pad + dilation * (S - 1) + output_pad + 1
 *    In this model, R = S = 3, stride = 2, pad = 1, dilation = 1, output_pad = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 
void ConvTranspose2d_old(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[1];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];
 
  const size_t stride = 2;
  const size_t pad = 1;
  const size_t dilation = 1;

  for (size_t oc = 0; oc < K; ++oc) {
    for (size_t oh = 0; oh < OH; ++oh) {
      for (size_t ow = 0; ow < OW; ++ow) {
        float o = bias->buf[oc];
        for (size_t c = 0; c < C; ++c) {
          for (size_t r = 0; r < R; ++r) {
            for (size_t s = 0; s < S; ++s) {
              if ((oh - (r * dilation - pad)) % stride != 0) continue;
              if ((ow - (s * dilation - pad)) % stride != 0) continue;
              size_t h = (oh - (r * dilation - pad)) / stride;
              size_t w = (ow - (s * dilation - pad)) / stride;
              if (h >= H || w >= W) continue;
              o += in->buf[c * H * W + h * W + w] * 
                weight->buf[c * K * R * S + oc * R * S + r * S + s];
            }
          }
        }
        out->buf[oc * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}
*/

__global__ void conv_transpose2d_kernel(float *I, float *W_, float *B, float *O, 
                                        int C, int H, int W, int K, int R, int S, 
                                        int OH, int OW, int pad, int stride, int dilation) {
    
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int oc = tidx / (OH * OW);
    const int oh = (tidx / OW) % OH;
    const int ow = tidx % OW;

    if (oc >= K || oh >= OH || ow >= OW) return;
  
    float sum = B[oc];  // bias 초기화

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                if ((oh - r * dilation + pad) % stride != 0) continue;
                if ((ow - s * dilation + pad) % stride != 0) continue;

                int h = (oh - r * dilation + pad) / stride;
                int w = (ow - s * dilation + pad) / stride;

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    sum += I[(c * H + h) * W + w] * W_[((c * K + oc) * R + r) * S + s];
                }
            }
        }
    }
    O[(oc * OH + oh) * OW + ow] = sum;
}

void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];
    size_t K = weight->shape[1];
    size_t R = weight->shape[2];
    size_t S = weight->shape[3];
    size_t OH = out->shape[2];
    size_t OW = out->shape[3];

    const size_t stride = 2;
    const size_t pad = 1;
    const size_t dilation = 1;

    float *I_gpu, *W_gpu, *B_gpu, *O_gpu;
    // CHECK_CUDA(cudaMalloc(&I_gpu, C * H * W * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&W_gpu, C * K * R * S * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&B_gpu, K * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&O_gpu, K * OH * OW * sizeof(float)));

    // CHECK_CUDA(cudaMemcpy(I_gpu, in->buf, C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(W_gpu, weight->buf, C * K * R * S * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(B_gpu, bias->buf, K * sizeof(float), cudaMemcpyHostToDevice));

    int total_threads = K * OH * OW;
    int block_size = 1024;
    dim3 blockDim(block_size);
    dim3 gridDim((total_threads + block_size - 1) / block_size);
    conv_transpose2d_kernel<<<gridDim, blockDim>>>(I_gpu, W_gpu, B_gpu, O_gpu, 
                                                   C, H, W, K, R, S, OH, OW, pad, stride, dilation);

    // CHECK_CUDA(cudaMemcpy(out->buf, O_gpu, K * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

    // CHECK_CUDA(cudaFree(I_gpu));
    // CHECK_CUDA(cudaFree(W_gpu));
    // CHECK_CUDA(cudaFree(B_gpu));
    // CHECK_CUDA(cudaFree(O_gpu));
}



/* BatchNorm2d (track_running_stats=False)
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C]
 * @param [in3]   bias: [C]
 * @param [out]    out: [N, C, H, W]  
 * 
 *    out = weight * (in - mean) / sqrt(var + 1e-5) + bias 
 * 
 * 'N' is the number of input tensors.
 * 'C' is the number of channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 
void BatchNorm2d_old(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];

  const float eps = 1e-5f;

  for (size_t c = 0; c < C; c++) {
    // 1. Caculate mean for each channel
    float mean = 0.0f;
    float var = 0.0f;
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        float val = in->buf[c * H * W + h * W + w];
        mean += val;
      }
    }
    mean /= (H * W);

    // 2. Caculate variance for each channel
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        float val = in->buf[c * H * W + h * W + w];
        var += (val - mean) * (val - mean);
      }
    }
    var /= (H * W);

    // 3. Normalize with the calculated mean and variance
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        out->buf[c * H * W + h * W + w] =
          weight->buf[c] * 
          (in->buf[c * H * W + h * W + w] - mean) /
          sqrt(var + eps) +
          bias->buf[c];
      }
    }
  }
}
*/

__global__ void BatchNorm2d_calc(float *in, float *out, float *weight, float *bias, int C, int W, int H) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return; // C보다 크면 리턴

    float mean = 0.0f;
    float var = 0.0f;

    // 1. Calculate mean for each channel
    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            float val = in[c * H * W + h * W + w];
            mean += val;
        }
    }
    mean /= (H * W);

    // 2. Calculate variance for each channel
    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            float val = in[c * H * W + h * W + w];
            var += (val - mean) * (val - mean);
        }
    }
    var /= (H * W);

    // Store mean and variance for further use in result kernel
    out[c * 2] = mean;      // mean
    out[c * 2 + 1] = var;   // variance
}

__global__ void BatchNorm2d_result(float *mean_var, float *in, float *out, float *weight, float *bias, int C, int W, int H) {
    const float eps = 1e-5f;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return; // C보다 크면 리턴

    float mean = mean_var[c * 2];
    float var = mean_var[c * 2 + 1];

    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            out[c * H * W + h * W + w] =
                weight[c] * (in[c * H * W + h * W + w] - mean) / sqrt(var + eps) + bias[c];
        }
    }
}

void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
    // CUDA 메모리 할당
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];

    float *d_in, *d_out, *d_weight, *d_bias, *d_mean_var;

    // 입력, 출력, 가중치, 바이어스, 평균 및 분산 할당
    // cudaMalloc(&d_in, C * H * W * sizeof(float));
    // cudaMalloc(&d_out, C * H * W * sizeof(float));
    // cudaMalloc(&d_weight, C * sizeof(float));
    // cudaMalloc(&d_bias, C * sizeof(float));
    // cudaMalloc(&d_mean_var, C * 2 * sizeof(float)); // mean과 variance 저장용

    // 호스트 메모리에서 디바이스 메모리로 데이터 복사
    // cudaMemcpy(d_in, in->buf, C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_weight, weight->buf, C * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bias, bias->buf, C * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA 블록 및 그리드 크기 설정
    size_t block_size = 1024;
    dim3 blockDim(block_size);
    dim3 gridDim((C + block_size - 1) / block_size);

    // 1. 평균과 분산 계산 커널 호출
    BatchNorm2d_calc<<<gridDim, blockDim>>>(d_in, d_mean_var, d_weight, d_bias, C, W, H);

    // 2. 정규화 및 결과 계산 커널 호출
    BatchNorm2d_result<<<gridDim, blockDim>>>(d_mean_var, d_in, d_out, d_weight, d_bias, C, W, H);

    // 결과를 호스트 메모리로 복사
    // cudaMemcpy(out->buf, d_out, C * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // // CUDA 메모리 해제
    // cudaFree(d_in);
    // cudaFree(d_out);
    // cudaFree(d_weight);
    // cudaFree(d_bias);
    // cudaFree(d_mean_var);
}





/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.

void LeakyReLU_CPU(Tensor *inout) {
  size_t N = inout->num_elem();

  const float alpha = 0.01;

  for (size_t i = 0; i < N; i++) {
    if (inout->buf[i] < 0) { inout->buf[i] *= alpha; }
  }
}
*/

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void LeakyReLU_kernel(float *inout, size_t N, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { 
    if (inout[idx] < 0) { inout[idx] *= alpha; }
  }
}  

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  const float alpha = 0.01;

  float *d_inout;

  // CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice));

  LeakyReLU_kernel<<<(N + 255) / 256, 256>>>(d_inout, N, alpha);
  //CHECK_CUDA(cudaDeviceSynchronize());

//   CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost));
//   CHECK_CUDA(cudaFree(d_inout));
 }

/* Conv2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [K, C, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *
 *   OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1
 *   OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 *   In this model, R = S = 3, stride = 1, pad = 1, dilation = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 
void Conv2d_old(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < K; oc++) {
      for (size_t oh = 0; oh < OH; oh++) {
        for (size_t ow = 0; ow < OW; ow++) {
          float o = bias->buf[oc];
          for (size_t c = 0; c < C; c++) {
            for (size_t r = 0; r < R; r++) {
              for (size_t s = 0; s < S; s++) {
                size_t h = oh * stride - pad + r * dilation;
                size_t w = ow * stride - pad + s * dilation;
                if (h >= H || w >= W) continue;
                o += in->buf[n * C * H * W + c * H * W + h * W + w] *
                  weight->buf[oc * C * R * S + c * R * S + r * S + s];
              }
            }
          }
          out->buf[n * K * OH * OW + oc * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
}
*/
__global__ void conv2d_kernel(float *I, float *W_, float *B, float *O, int N, int C,
                              int H, int W, int K, int R, int S, int OH, int OW,
                              int pad, int stride, int dilation) {
  
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int on = tidx / (K * OH * OW);
  const int oc = (tidx / (OH * OW)) % K;
  const int oh = (tidx / OW) % OH;
  const int ow = tidx % OW;

  if (on >= N || oc >= K || oh >= OH || ow >= OW) return;
  
  float sum = B[oc];  // bia 초기화
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int h = oh * stride - pad + r * dilation;
        int w = ow * stride - pad + s * dilation;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          sum += I[((on * C + c) * H + h) * W + w] *
                 W_[((oc * C + c) * R + r) * S + s];
        }
      }
    }
  }
  O[((on * K + oc) * OH + oh) * OW + ow] = sum;
}

void Conv2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;
  
  float *I_gpu, *W_gpu, *B_gpu, *O_gpu;
  // CHECK_CUDA(cudaMalloc(&I_gpu, N * C * H * W * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&W_gpu, K * C * R * S * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&B_gpu, K * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&O_gpu, N * K * OH * OW * sizeof(float)));

  // CHECK_CUDA(cudaMemcpy(I_gpu, in->buf, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(W_gpu, weight->buf, K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(B_gpu, bias->buf, K * sizeof(float), cudaMemcpyHostToDevice));

  int total_threads = N * K * OH * OW;
  int block_size = 1024;
  dim3 blockDim(block_size);
  dim3 gridDim((total_threads + block_size - 1) / block_size);
  conv2d_kernel<<<gridDim, blockDim>>>(I_gpu, W_gpu, B_gpu, O_gpu, N, C, H, W, K,
                                       R, S, OH, OW, pad, stride, dilation);

  // CHECK_CUDA(cudaMemcpy(out->buf, O_gpu, N * K * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

  // CHECK_CUDA(cudaFree(I_gpu));
  // CHECK_CUDA(cudaFree(W_gpu));
  // CHECK_CUDA(cudaFree(B_gpu));
  // CHECK_CUDA(cudaFree(O_gpu));
}

/* Tanh 
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
// void Tanh(Tensor *inout) {
//   size_t N = inout->num_elem();


// //버프를 호출 하는 애를 
//   for (size_t i = 0; i < N; i++) {
//     inout->buf[i] = tanh(inout->buf[i]);
//   }
// }


// CUDA 커널 함수
__global__ void TanhKernel(float *buf, size_t N) {
    // 현재 스레드의 인덱스를 계산
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 텐서 크기(N) 내의 인덱스만 처리
    if (i < N) {
        buf[i] = tanhf(buf[i]); // CUDA에서 사용하는 tanhf 함수 (float용)
    }
}

// 호스트 함수
void Tanh(Tensor *inout) {
    size_t N = inout->num_elem();
    float *gpu;

    // // GPU 메모리 할당
    // cudaMalloc(&d_buf, N * sizeof(float));

    // // 데이터를 GPU로 복사
    // cudaMemcpy(d_buf, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);

    // 스레드와 블록의 크기 정의
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // CUDA 커널 호출
    TanhKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu, N);

    // 연산 결과를 다시 호스트 메모리로 복사
    // cudaMemcpy(inout->buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);

    // // GPU 메모리 해제
    //CHECK_CUDA(cudaFree(gpu));
}