/*
 * Last update: 10/02/2024
 * Issue date:  10/02/2024
 *
 * Copyright (C) 2024, Smiril <sonar@gmx.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
*/

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

/*
 * FIPS 180-2 SHA-224/256/384/512 implementation
 * Last update: 02/02/2007
 * Issue date:  04/30/2005
 *
 * Copyright (C) 2013, Con Kolivas <kernel@kolivas.org>
 * Copyright (C) 2005, 2007 Olivier Gay <olivier.gay@a3.epfl.ch>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

// System includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#define MAX_LINE_LENGTH 42

#include <assert.h>
#include <cuda.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

#define PWD_LEN 40
FILE *file1;
FILE *file2;
char pwd[sizeof(char)*(PWD_LEN + 1)];
////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N = 1048576 * 32;

////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA Sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
#define THREADS_PER_BLOCK 256
#if __CUDA_ARCH__ >= 200
#define MY_KERNEL_MAX_THREADS (2 * THREADS_PER_BLOCK)
#define MY_KERNEL_MIN_BLOCKS 3
#else
#define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK
#define MY_KERNEL_MIN_BLOCKS 2
#endif

__global__ static void __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS) reduceKernel(float *d_Result, float *d_Input, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadN = gridDim.x * blockDim.x;
  float sum = 0;

  for (int pos = tid; pos < N; pos += threadN) sum += d_Input[pos];
  __syncthreads();

  d_Result[tid] = sum;
}

char *password_good;  //this changed only once, when we found the good passord
char *password; //this contains the actual password
char hfile[255];    //the hashes file name
long counter = 0;    //this couning probed passwords
int finished = 0;
int flag = 0;
pthread_barrier_t barr;

#define UNPACK32(x, str)                      \
{                                             \
    *((str) + 3) = (uint8_t) ((x)      );       \
    *((str) + 2) = (uint8_t) ((x) >>  8);       \
    *((str) + 1) = (uint8_t) ((x) >> 16);       \
    *((str) + 0) = (uint8_t) ((x) >> 24);       \
}

#define PACK32(str, x)                        \
{                                             \
    *(x) =   ((uint32_t) *((str) + 3)      )    \
           | ((uint32_t) *((str) + 2) <<  8)    \
           | ((uint32_t) *((str) + 1) << 16)    \
           | ((uint32_t) *((str) + 0) << 24);   \
}

#define SHA256_SCR(i)                         \
{                                             \
    w[i] =  SHA256_F4(w[i -  2]) + w[i -  7]  \
          + SHA256_F3(w[i - 15]) + w[i - 16]; \
}

uint32_t sha256_h0[8] =
            {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

uint32_t sha256_k[64] =
            {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
             0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
             0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
             0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
             0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
             0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
             0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
             0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
             0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

/* SHA-256 functions */

void sha256_transf(sha256_ctx *ctx, const unsigned char *message,unsigned int block_nb)
{
    uint32_t w[64];
    uint32_t wv[8];
    uint32_t t1, t2;
    const unsigned char *sub_block;
    int i;

    int j;

    for (i = 0; i < (int) block_nb; i++) {
        sub_block = message + (i << 6);

        for (j = 0; j < 16; j++) {
            PACK32(&sub_block[j << 2], &w[j]);
        }

        for (j = 16; j < 64; j++) {
            SHA256_SCR(j);
        }

        for (j = 0; j < 8; j++) {
            wv[j] = ctx->h[j];
        }

        for (j = 0; j < 64; j++) {
            t1 = wv[7] + SHA256_F2(wv[4]) + CH(wv[4], wv[5], wv[6])
                + sha256_k[j] + w[j];
            t2 = SHA256_F1(wv[0]) + MAJ(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }

        for (j = 0; j < 8; j++) {
            ctx->h[j] += wv[j];
        }
    }
}

void sha256_init(sha256_ctx *ctx)
{
    int i;
    for (i = 0; i < 8; i++) {
        ctx->h[i] = sha256_h0[i];
    }

    ctx->len = 0;
    ctx->tot_len = 0;
}

void sha256_update(sha256_ctx *ctx, const unsigned char *message,unsigned int len)
{
    unsigned int block_nb;
    unsigned int new_len, rem_len, tmp_len;
    const unsigned char *shifted_message;

    tmp_len = SHA256_BLOCK_SIZE - ctx->len;
    rem_len = len < tmp_len ? len : tmp_len;

    memcpy(&ctx->block[ctx->len], message, rem_len);

    if (ctx->len + len < SHA256_BLOCK_SIZE) {
        ctx->len += len;
        return;
    }

    new_len = len - rem_len;
    block_nb = new_len / SHA256_BLOCK_SIZE;

    shifted_message = message + rem_len;

    sha256_transf(ctx, ctx->block, 1);
    sha256_transf(ctx, shifted_message, block_nb);

    rem_len = new_len % SHA256_BLOCK_SIZE;

    memcpy(ctx->block, &shifted_message[block_nb << 6],
           rem_len);

    ctx->len = rem_len;
    ctx->tot_len += (block_nb + 1) << 6;
}

void sha256_final(sha256_ctx *ctx, unsigned char *digest)
{
    unsigned int block_nb;
    unsigned int pm_len;
    unsigned int len_b;

    int i;

    block_nb = (1 + ((SHA256_BLOCK_SIZE - 9)
                     < (ctx->len % SHA256_BLOCK_SIZE)));

    len_b = (ctx->tot_len + ctx->len) << 3;
    pm_len = block_nb << 6;

    memset(ctx->block + ctx->len, 0, pm_len - ctx->len);
    ctx->block[ctx->len] = 0x80;
    UNPACK32(len_b, ctx->block + pm_len - 4);

    sha256_transf(ctx, ctx->block, block_nb);

    for (i = 0 ; i < 8; i++) {
        UNPACK32(ctx->h[i], &digest[i << 2]);
    }
}

void sha256(const unsigned char *message, unsigned int len, unsigned char *digest)
{
    sha256_ctx ctx;

    sha256_init(&ctx);
    sha256_update(&ctx, message, len);
    sha256_final(&ctx, digest);
}

char *nextpass() {
    char line[MAX_LINE_LENGTH * sizeof(char*)];
    
    while (fgets(line, MAX_LINE_LENGTH, file2) != NULL) {
        line[strcspn(line, "\n")] = '\0';
        strcpy(pwd, line);
    }

    return pwd;
}

void status_thread(void) {
    int pwds;

    const short status_sleep = 5;
    while(1) {
        sleep(status_sleep);
        pwds = counter / status_sleep;
        counter = 0;

        if (finished != 0 && feof(file1)) {
            break;
        }
        
        printf("Probing: '%hs' [%hd pwds/sec]\n", password, pwds);
        }
}

void crack_thread(void) {
    char *current = (char*)malloc(MAX_LINE_LENGTH);
    char line1[MAX_LINE_LENGTH];
    //char cur[SHA256_DIGEST_SIZE];
    //char lane2[SHA256_DIGEST_SIZE];
    char hashed_password[SHA256_DIGEST_SIZE]; // Each byte of hash produces two characters in hex
    file2 = fopen("/usr/local/share/rockyou.txt", "r");
    pthread_mutex_t mutex;
    pthread_mutex_lock(&mutex);
    flag=1;

    while (flag == 1) {
        current = nextpass();
        file1 = fopen(hfile, "r");
        while (!feof(file1)) {
            fgets(line1, MAX_LINE_LENGTH, file1);
            line1[strcspn(line1, "\n")] = '\0';
                
            sha256((const unsigned char *)current, (unsigned int)strlen(current), (unsigned char *)hashed_password);
                /*
            for (int i = 0; i < SHA256_DIGEST_SIZE; i++) {
                    sprintf(lane2,"%02x", (unsigned char)hashed_password[i]);
                    strcat(cur,lane2);
                }
            */
            if (strcmp(hashed_password,line1) == 0) {
                    strcpy(password_good, current);
                    finished = 1;
		    printf("GOOD: password cracked: '%s'\n", password_good);
		    free((void *)password_good);
                    break;
                }
        }
        
        counter++;
        
        if (finished != 0 && feof(file1)) {
            flag = 0;
            break;
        }
        
        free((void *)current);
    }
    pthread_mutex_unlock (&mutex);
    fclose(file1);
    fclose(file2);
}

void crack_start(unsigned int threads) {
    pthread_t th[101];
    unsigned int i;

    pthread_barrier_init(&barr, NULL, threads);
    int res = pthread_barrier_wait(&barr);
    if(res == PTHREAD_BARRIER_SERIAL_THREAD) {
    
    for (i = 0; i < threads; i++) {
        (void) pthread_create(&th[i], NULL, (void *(*)(void *))crack_thread, NULL);
    }

    (void) pthread_create(&th[100], NULL, (void *(*)(void *))status_thread, NULL);

    for (i = 0; i < threads; i++) {
        (void) pthread_join(th[i], NULL);
    }

    (void) pthread_join(th[100], NULL);

    } else if(res != 0) {
        perror("Threading");
    } else {
        (void) pthread_create(&th[1], NULL, (void *(*)(void *))crack_thread, NULL);

        (void) pthread_create(&th[10], NULL, (void *(*)(void *))status_thread, NULL);

        (void) pthread_join(th[1], NULL);

        (void) pthread_join(th[10], NULL);
       }
}

int init(int threadsx, char *mir) {
    int threads = 1;
    threads = threadsx;
    if (threads < 1) threads = 1;
    if (threads > 100) {
        printf("INFO: number of threads adjusted to 12%s","\n");
        threads = 12;
    }
    strcpy((char*)&hfile, mir);
    crack_start(threads);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    // Print author
    printf("shaCrack! 0.2 by Smiril (sonar@gmx.com)\n\n");
    int s;
    int help = 0;
    int threads = 1;
    char *hash;

    if (argc == 1) {
        printf("USAGE: %s  --threads [NUM] --hash [hashes.ext]\n",argv[0]);
        printf("       For more information please run \"%s --help\"\n",argv[0]);
        help = 1;
    } else {
        for (s = 1; s < argc; s++) {
            if (strcmp(argv[s],"--help") == 0) {
                printf("Usage:   %s  --threads [NUM] --hash [hashes.ext]\n\n",argv[0]);
                printf("Options: --help: show this screen.%s","\n");
                printf("         --threads: you can specify how many threads%s","\n");
                printf("                    will be run, maximum 100 (default: 12)\n%s","\n");
		printf("         --hash: you can specify hash file%s","\n");
                printf("Info:    This program supports only ASCII HASH FILES.%s","\n");
                help = 1;
                break;
            } else if (strcmp(argv[s],"--threads") == 0) {
                if ((s + 1) < argc) {
                    sscanf(argv[++s], "%d", &threads);
                    if (threads < 1) threads = 1;
                    if (threads > 100) {
                        printf("INFO: number of threads adjusted to 12\n");
                        threads = 12;
                    } 
		} else {
                    printf("ERROR: missing parameter for option: --threads %s","\n");
                    help = 1;
        	}
           } else if (strcmp(argv[s],"--hash") == 0) {
                if ((s + 1) < argc) {
                    sscanf(argv[++s], "%s", &hash);
                } else {
                    printf("ERROR: missing parameter for option: --hash %s","\n");
                    help = 1;
                }
            } else {
                printf("%s","\n");
            }
        }
    }

    if (help == 1) {
        return;
    }

  // Solver config
  TGPUplan plan[MAX_GPU_COUNT];

  // GPU reduction results
  float h_SumGPU[MAX_GPU_COUNT];

  float sumGPU;
  double sumCPU, diff;

  int i, j, gpuBase, GPU_N, deviceID;

  cudaDeviceProp Propx;

  printf("Starting %s\n",argv[0]);
  checkCudaErrors(cudaGetDeviceCount(&GPU_N));
  checkCudaErrors(cudaGetDevice(&deviceID));
  checkCudaErrors(cudaGetDeviceProperties(&Propx, deviceID));

  int threadsPerBlock = (Propx.major >= 2 ? 2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK);

  const int BLOCK_N = 32;
  const int THREAD_N = threadsPerBlock;
  const int ACCUM_N = BLOCK_N *THREAD_N;

  if (GPU_N > MAX_GPU_COUNT) {
    GPU_N = MAX_GPU_COUNT;
  }

  printf("CUDA-capable device count: %i\n", GPU_N);

  printf("Generating input data...\n%s","\n");


  // Subdividing input data across GPUs
  // Get data sizes for each GPU
  for (i = 0; i < GPU_N; i++) {
    plan[i].dataN = DATA_N / GPU_N;
  }

  // Take into account "odd" data sizes
  for (i = 0; i < DATA_N % GPU_N; i++) {
    plan[i].dataN = init(threads,hash);
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
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024));
    checkCudaErrors(cudaStreamCreate(&plan[i].stream));
    // Allocate memory
    checkCudaErrors(cudaMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&plan[i].h_Sum_from_device, ACCUM_N * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&plan[i].h_Data,plan[i].dataN * sizeof(float)));

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
    checkCudaErrors(cudaMemcpyAsync(plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream));

    // Perform GPU computations
    reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
    getLastCudaError("reduceKernel() execution failed.\n");
    cudaDeviceSynchronize();
    // Read back GPU results
    checkCudaErrors(cudaMemcpyAsync(plan[i].h_Sum_from_device, plan[i].d_Sum,ACCUM_N * sizeof(float),cudaMemcpyDeviceToHost, plan[i].stream));
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
  StopWatchInterface *timerc = NULL;
  sdkCreateTimer(&timerc);

  // start the timer
  sdkStartTimer(&timerc);
  // Compute on Host CPU
  printf("Computing with Host CPU...%s","\n");

  sumCPU = 0;

  for (i = 0; i < GPU_N; i++) {
    for (j = 0; j < plan[i].dataN; j++) {
      sumCPU += plan[i].h_Data[j];
    }
  }
  sdkStopTimer(&timerc);
  printf("  CPU Processing time: %f (ms)\n\n", sdkGetTimerValue(&timerc));
  sdkDeleteTimer(&timerc);

  // Compare GPU and CPU results
  printf("Comparing GPU and Host CPU results...%s","\n");
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
