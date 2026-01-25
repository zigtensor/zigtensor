#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int num_elements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num_elements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

int main(void) {
  cudaError_t err = cudaSuccess;
  int num_elements = 10000;

  size_t size = num_elements * sizeof(float);

  printf("[Vector addition of %d elements]\n", num_elements);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "[Could not allocate vectors on host]\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_elements; i++) {
    h_A[i] = rand() / float(RAND_MAX);
    h_B[i] = rand() / float(RAND_MAX);
  }

  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Could not allocate vector A on device. Error: %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Could not allocate vector B on device. Error: %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Could not allocate vector C on device. Error: %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("[Copy input data from host to device]\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Could not copy vector A to device. Error: %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Could not copy vector B to device. Error: %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
  printf("[Kernel launch with %d blocks of %d threads]\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[Failed to launch vectorAdd kernel. Error: %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("[Bring output back to host]\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Failed to copy back to host. Errors %s]\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_elements; i++) {
    if (fabs(h_B[i] + h_A[i] - h_C[i]) > 1e-5) {
      fprintf(stderr,
              "[Verification failed on element, %d, with A value %f, B value "
              "%f, and C value %f",
              i, h_A[i], h_B[i], h_C[i]);
      exit(EXIT_FAILURE);
    }
  }

  printf("[PASSED]\n");

  err = cudaFree(d_A);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Failed to free A from device. Error %s]\n",
            cudaGetErrorString(err));
  }

  err = cudaFree(d_B);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Failed to free B from device. Error %s]\n",
            cudaGetErrorString(err));
  }

  err = cudaFree(d_C);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Failed to free C from device. Error %s]\n",
            cudaGetErrorString(err));
  }

  free(h_A);
  free(h_B);
  free(h_C);

  printf("[DONE]\n");
  return 0;
}
