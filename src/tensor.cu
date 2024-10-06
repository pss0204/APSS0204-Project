#include "model.h"

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
CHECK_CUDA( cudaMalloc(&d_buf,N_*sizeof(float)));//pss


  // cudaMalloc
}

Tensor::Tensor(const vector<size_t> &shape_, float *d_buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  // cudaMalloc 
  CHECK_CUDA(cudaMalloc(&d_buf,N_ * sizeof(float)));//pss
  //memcpy(buf, buf_, N_ * sizeof(float));
  // cudaMemcpy
  CHECK_CUDA(cudaMemcpy(d_buf, d_buf_, N_ * sizeof(float),cudaMemcpyHostToDevice));//pss
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
  if (d_buf != nullptr) CHECK_CUDA(cudaFree(d_buf));
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}