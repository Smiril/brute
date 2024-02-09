/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on
 * performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

// System includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#define MAX_LINE_LENGTH 40
#include <openssl/sha.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

#define PWD_LEN 40
    FILE *file1;
    FILE *file2;
    char pwd[sizeof(char)*(PWD_LEN + 1)];
    char *current;
////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
char *fir;
const int MAX_GPU_COUNT = 32;
const int DATA_N = 1048576 * 32;

////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA Sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
__global__ static void reduceKernel(float *d_Result, float *d_Input, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadN = gridDim.x * blockDim.x;
  float sum = 0;

  for (int pos = tid; pos < N; pos += threadN) sum += d_Input[pos];

  d_Result[tid] = sum;
}
char password_good[40] = {'\0', '\0'};  //this changed only once, when we found the good passord
char password[40+1] = {'\0','\0'}; //this contains the actual password
char hfile[255];    //the hashes file name
long counter = 0;    //this couning probed passwords
int finished = 0;

void sha256(const char *input, char *output) {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input, strlen(input));
    SHA256_Final((unsigned char*)output, &sha256);
}

char *nextpass() {
    char line[MAX_LINE_LENGTH * sizeof(char*)];
    
    while (fgets(line, MAX_LINE_LENGTH, file2) != NULL) {
        line[strcspn(line, "\n")] = '\0';
        strcpy(pwd, line);
    }

    return pwd;
}

void status_thread() {
    int pwds;

    const short status_sleep = 1;
    while(1) {
        sleep(status_sleep);
        pwds = counter / status_sleep;
        counter = 0;

        if (finished != 0) {
            break;
        }
        
        printf("Probing: '%s' [%d pwds/sec]\n", password, pwds);
        }
}

char *crack_thread() {
    char line1[MAX_LINE_LENGTH];
    char cur[SHA256_DIGEST_LENGTH];
    char lane2[SHA256_DIGEST_LENGTH];
    char hashed_password[SHA256_DIGEST_LENGTH * 2 + 1]; // Each byte of hash produces two characters in hex
    file2 = fopen("/usr/local/share/rockyou.txt", "r");
    while (1) {
        current = nextpass();
        file1 = fopen(hfile, "r");
        while (!feof(file1)) {
            fgets(line1, MAX_LINE_LENGTH, file1);
            line1[strcspn(line1, "\n")] = '\0';
                
            sha256(current, hashed_password);
                
            for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
                    sprintf(lane2,"%02x", (unsigned char)hashed_password[i]);
                    strcat(cur,lane2);
                }
            
            if (strcmp(cur,line1)) {
                    strcpy(password_good, current);
                    finished = 1;
                    return password_good;
                    break;
                }
        }
        
        counter++;
        
        if (finished != 0) {
            break;
        }
        
        free(current);
    }
    fclose(file1);
    fclose(file2);
    return password_good;
}


void crack_start(unsigned int threads) {
    pthread_t th[101];
    unsigned int i;

    for (i = 0; i < threads; i++) {
        (void) pthread_create(&th[i], NULL, (void *(*)(void *))crack_thread, NULL);
    }

    (void) pthread_create(&th[100], NULL, (void *(*)(void *))status_thread, NULL);

    for (i = 0; i < threads; i++) {
        (void) pthread_join(th[i], NULL);
    }

    (void) pthread_join(th[100], NULL);
}

int init(int threadsx, char *mir) {
    int threads = 1;
    threads = threadsx;
    strcpy((char*)&hfile, mir);
    crack_start(threads);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  if (argc < 2) {
        printf("USAGE: %s hashes.ext\n",argv[0]);
	exit(1);
    }
  // Solver config
  TGPUplan plan[MAX_GPU_COUNT];

  // GPU reduction results
  float h_SumGPU[MAX_GPU_COUNT];

  float sumGPU;
  double sumCPU, diff;

  int i, j, gpuBase, GPU_N;

  const int BLOCK_N = 32;
  const int THREAD_N = 256;
  const int ACCUM_N = BLOCK_N * THREAD_N;

  printf("Starting simpleMultiGPU\n");
  checkCudaErrors(cudaGetDeviceCount(&GPU_N));

  if (GPU_N > MAX_GPU_COUNT) {
    GPU_N = MAX_GPU_COUNT;
  }

  printf("CUDA-capable device count: %i\n", GPU_N);

  printf("Generating input data...\n\n");


  // Subdividing input data across GPUs
  // Get data sizes for each GPU
  for (i = 0; i < GPU_N; i++) {
    plan[i].dataN = DATA_N / GPU_N;
  }

  // Take into account "odd" data sizes
  for (i = 0; i < DATA_N % GPU_N; i++) {
    plan[i].dataN = init(100,argv[1]);
  }

  // Assign data ranges to GPUs
  gpuBase = 0;

  for (i = 0; i < GPU_N; i++) {
    plan[i].h_Sum = h_SumGPU + i;
    gpuBase += plan[i].dataN;
  }

  // Create streams for issuing GPU command asynchronously and allocate memory
  // (GPU and System page-locked)
  for (i = 0; i < GPU_N; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaStreamCreate(&plan[i].stream));
    // Allocate memory
    checkCudaErrors(
        cudaMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float)));
    checkCudaErrors(
        cudaMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&plan[i].h_Sum_from_device,
                                   ACCUM_N * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&plan[i].h_Data,
                                   plan[i].dataN * sizeof(float)));

    for (j = 0; j < plan[i].dataN; j++) {
      plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
    }
  }

  // Start timing and compute on GPU(s)
  printf("Computing with %d GPUs...\n", GPU_N);
  // create and start timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  // start the timer
  sdkStartTimer(&timer);

  // Copy data to GPU, launch the kernel and copy data back. All asynchronously
  for (i = 0; i < GPU_N; i++) {
    // Set device
    checkCudaErrors(cudaSetDevice(i));

    // Copy input data from CPU
    checkCudaErrors(cudaMemcpyAsync(plan[i].d_Data, plan[i].h_Data,
                                    plan[i].dataN * sizeof(float),
                                    cudaMemcpyHostToDevice, plan[i].stream));

    // Perform GPU computations
    reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
    getLastCudaError("reduceKernel() execution failed.\n");

    // Read back GPU results
    checkCudaErrors(cudaMemcpyAsync(plan[i].h_Sum_from_device, plan[i].d_Sum,
                                    ACCUM_N * sizeof(float),cudaMemcpyDeviceToHost, plan[i].stream));
  }

  // Process GPU results
  for (i = 0; i < GPU_N; i++) {
    float sum;

    // Set device
    checkCudaErrors(cudaSetDevice(i));

    // Wait for all operations to finish
    cudaStreamSynchronize(plan[i].stream);

    // Finalize GPU reduction for current subvector
    sum = 0;

    for (j = 0; j < ACCUM_N; j++) {
      sum += plan[i].h_Sum_from_device[j];
    }

    *(plan[i].h_Sum) = (float)sum;

    // Shut down this GPU
    checkCudaErrors(cudaFreeHost(plan[i].h_Sum_from_device));
    checkCudaErrors(cudaFree(plan[i].d_Sum));
    checkCudaErrors(cudaFree(plan[i].d_Data));
    checkCudaErrors(cudaStreamDestroy(plan[i].stream));
  }

  sumGPU = 0;

  for (i = 0; i < GPU_N; i++) {
    sumGPU += h_SumGPU[i];
  }

  sdkStopTimer(&timer);
  printf("  GPU Processing time: %f (ms)\n\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  // Compute on Host CPU
  printf("Computing with Host CPU...\n\n");

  sumCPU = 0;

  for (i = 0; i < GPU_N; i++) {
    for (j = 0; j < plan[i].dataN; j++) {
      sumCPU += plan[i].h_Data[j];
    }
  }

  // Compare GPU and CPU results
  printf("Comparing GPU and Host CPU results...\n");
  diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
  printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
  printf("  Relative difference: %E \n\n", diff);


  // Cleanup and shutdown
  for (i = 0; i < GPU_N; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaFreeHost(plan[i].h_Data));
  }

  return EXIT_SUCCESS;
}
