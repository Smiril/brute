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

// Standard headers
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#define MAX_LINE_LENGTH 42
#define PWD_LEN 42
FILE *file1;
FILE *file2;
char pwd[MAX_LINE_LENGTH * sizeof(char*)];
sem_t mutex;
sem_t barrier;

int count;
int n=1;
#include "bruteforce.h"

char password_good[MAX_LINE_LENGTH * sizeof(char*)] = {'\0','\0'};  //this changed only once, when we found the good passord
char password[MAX_LINE_LENGTH * sizeof(char*)] = {'\0','\0'}; //this contains the actual password
char hfile[255];    //the hashes file name
long counter = 0;    //this counting probed passwords
int finished = 0;
int flag = 0;

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
    
    file2 = fopen("/usr/local/share/brute/rockyou.txt", "r");

    if(file2 == NULL) {
	perror("Wordlist");
	}
	
    while(fgets(line, MAX_LINE_LENGTH, file2) != NULL) {
        line[strcspn(line, "\n")] = '\0';
        sprintf(password,"%s",line);
        //strcpy(password, line);
	counter = counter + 1;
        //return password;
    }
	
	return password;
}

void status_thread() {
    int pwds;

    const short status_sleep = 1;
    while(1) {
        sleep(status_sleep);
        pwds = counter / status_sleep;

        if (finished != 0) {
            break;
        }
        
        printf("Probing: '%s' [%d pwds/sec]\n", password, pwds);
        }
}

void crack_thread(void) {
    char *current = (char*)malloc(MAX_LINE_LENGTH);
    char line1[MAX_LINE_LENGTH];
    //char cur[SHA256_DIGEST_SIZE];
    //char lane2[SHA256_DIGEST_SIZE];
    char hashed_password[SHA256_DIGEST_SIZE]; // Each byte of hash produces two characters in hex
	
    flag=1;
    sem_wait(&mutex);
    count=count + 1;
    sem_post(&mutex);

    if(count==n)
	sem_post(&barrier);
    sem_wait(&barrier);
    sem_post(&barrier);           // turnstile: a wait and a post occurs in  
 	
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
                    sprintf(password_good,"%s", current);
                    finished = 1;
		    printf("GOOD: \npassword %s \ncracked: '%s'\n", hashed_password, password_good);
		    //free((void *)password_good);
                    break;
                }
        }
        
        //counter++;
        
        if (finished != 0 && feof(file1)) {
              flag = 0;
	      break;
        }
        
        free((void *)current);
    }
    fclose(file1);
    fclose(file2);
}


void crack_start(unsigned int threads) {
    pthread_t th[101];
    unsigned int i;
    //sem_init(&mutex,0,1);
    //sem_init(&barrier,0,0);

    for (i = 0; i < threads; i++) {
        (void) pthread_create(&th[i], NULL, (void *(*)(void *))crack_thread, NULL);
    }

    (void) pthread_create(&th[100], NULL, (void *(*)(void *))status_thread, NULL);

    for (i = 0; i < threads; i++) {
        (void) pthread_join(th[i], NULL);
    }

    (void) pthread_join(th[100], NULL);   
}

int main(int argc, char **argv) {
    // Print author
    printf("shaCrack! 0.1 by Smiril (sonar@gmx.com)\n\n");
    clock_t start, finish;
    double  laufzeit;

    start = clock();
    int s, i, j;
    int help = 0;
    int threads = 0;

    if (argc < 2) {
        printf("USAGE: %s  --threads [NUM] --hash [hashes.ext]\n",argv[0]);
        printf("       For more information please run \"%s --help\"\n",argv[0]);
        help = 1;
    } else {
        for (s = 1; s < argc; s++) {
            if (strcmp(argv[s],"--help") == 0) {
                printf("Usage:   %s  --threads [NUM] --hash [hashes.ext]\n\n",argv[0]);
                printf("Options: --help: show this screen.\n");
                printf("         --threads: you can specify how many threads\n");
                printf("                    will be run, maximum 100 (default: 12)\n");
		printf("         --hash: you can specify hash file\n");
                printf("Info:    This program supports only ASCII HASH FILES.\n");
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
                    printf("ERROR: missing parameter for option: --threads\n");
                    help = 1;
                }
            } else if (strcmp(argv[s],"--hash") == 0) {
                if ((s + 1) < argc) {
                    sscanf(argv[++s], "%s", hfile);
                } else {
                    printf("ERROR: missing parameter for option: --hash\n");
                    help = 1;
                }
            } else {
                printf("\n");
            }
        }
    }

    if (help == 1) {
        return EXIT_SUCCESS;
    } else {
    	crack_start(threads);
    }
    finish = clock();

    laufzeit = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("  CPU Processing time: %2.6f (sec)\n\n", laufzeit);
    return EXIT_SUCCESS;
}
