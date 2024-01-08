/*
 * Last update: 07/01/2024
 * Issue date:  07/01/2024
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
#include <stdint.h>
#include <unistd.h>
#define MAX_LINE_LENGTH 4096
#include <openssl/sha.h>
char password[4096+1] = {'\0','\0'}; //this contains the actual password
char password_good[4096] = {'\0', '\0'};  //this changed only once, when we found the good passord
long counter = 0;    //this couning probed passwords
char hfile[255];    //the hashes file name
char statname[259];    //status xml file name filename + ".xml"
int finished = 0;


void crack_start(unsigned int threads);

void sha256(const char *input, char *output) {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input, strlen(input));
    SHA256_Final((unsigned char*)output, &sha256);
}

char *nextpass() {
    char *line = malloc(MAX_LINE_LENGTH * sizeof(char));
    char **Con  = malloc(MAX_LINE_LENGTH * sizeof(char*));
    int nCon = 0;
    FILE *file2;
    
    file2 = fopen("/usr/local/share/brute/rockyou.txt", "r");
    
    while (fgets(line, MAX_LINE_LENGTH, file2) != NULL) {
        if (! feof(file2)) {
            int len = strlen(line) + 1;
            Con[nCon] = malloc(len * sizeof(char));
            strcpy(Con[nCon], line);
            nCon++;
        }
    }
    fclose(file2);

    return *Con[nCon];
}

void *status_thread() {
    int pwds;
    const short status_sleep = 3;
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

void *crack_thread() {
    char *current;
    char ret[200];
    char cmd[100];
    FILE *file1;
    char line1[MAX_LINE_LENGTH];
    char cur[SHA256_DIGEST_LENGTH];
    char lane2[SHA256_DIGEST_LENGTH];
    char hashed_password[SHA256_DIGEST_LENGTH * 2 + 1]; // Each byte of hash produces two characters in hex
    
    while (1) {
        current = nextpass();
        file1 = fopen(hfile, "r");
        while (! feof(file1)) {
            fgets(line1, MAX_LINE_LENGTH, file1);
            line1[strcspn(line1, "\n")] = '\0';
                
            sha256(current, hashed_password);
                
            for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
                    sprintf(lane2,"%02x", (unsigned char)hashed_password[i]);
                    strcat(cur,lane2);
                }
            
            if (strcmp(cur,line1) != 0) {
                    strcpy(password_good, current);
                    finished = 1;
                    printf("GOOD: password cracked: '%s'\n", current);
                    break;
                }
        }

        fclose(file1);
        
        counter++;
        
        if (finished != 0) {
            break;
        }
        
        free(current);
    }
    //return 0;
}


void crack_start(unsigned int threads) {
    pthread_t th[101];
    unsigned int i;

    for (i = 0; i < threads; i++) {
        (void) pthread_create(&th[i], NULL, (void *)crack_thread, NULL);
    }

    (void) pthread_create(&th[100], NULL, (void *)status_thread, NULL);

    for (i = 0; i < threads; i++) {
        (void) pthread_join(th[i], NULL);
    }

    (void) pthread_join(th[100], NULL);
}

void init(int argc, char **argv) {
    int i, j;
    int help = 0;
    int threads = 1;

    if (argc == 1) {
        printf("USAGE: brute  [--threads NUM] hashes.ext\n");
        printf("       For more information please run \"brute --help\"\n");
        help = 1;
    } else {
        for (i = 1; i < argc; i++) {
            if (strcmp(argv[i],"--help") == 0) {
                printf("Usage:   brute  [--threads NUM] hashes.ext\n\n");
                printf("Options: --help: show this screen.\n");
                printf("         --threads: you can specify how many threads\n");
                printf("                    will be run, maximum 100 (default: 10)\n\n");
                printf("Info:    This program supports only TXT HASH FILES.\n");
                help = 1;
                break;
            } else if (strcmp(argv[i],"--threads") == 0) {
                if ((i + 1) < argc) {
                    sscanf(argv[++i], "%d", &threads);
                    if (threads < 1) threads = 1;
                    if (threads > 100) {
                        printf("INFO: number of threads adjusted to 10\n");
                        threads = 10;
                    }
                } else {
                    printf("ERROR: missing parameter for option: --threads!\n");
                    help = 1;
                }
            } else {
                strcpy((char*)&hfile, argv[i]);
            }
        }
    }

    if (help == 1) {
        return;
    }

    crack_start(threads);
}

int main(int argc, char **argv) {
    // Print author
    printf("shaCrack! 0.1 by Smiril (sonar@gmx.com)\n\n");

    init(argc,argv);
    
    return 0;
}
