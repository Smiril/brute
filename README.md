# brute-CUDA

brute - port for CUDA (works on Linux)

# known bugs
 - warnings + errors

Originally source code by Copyright (C) 2024 Smiril - sonar@gmx.com

Parts Copyright (C) by Other (see header)


## Compile

```shell
sudo make
```
## Usage
```shell
$ ./simpleMultiGPU --threads 2 --hash hashes.txt
```

## Example

 **sha256**
```shell
$ ./simpleMultiGPU --threads 1 --hash test.txt
shaCrack! 0.2 by Smiril (sonar@gmx.com)

Starting ./simpleMultiGPU
CUDA-capable device count: 1
Generating input data...

Computing with 1 GPUs...
  GPU Processing time: 11.488000 (ms)

Computing with Host CPU...
  CPU Processing time: 91.248001 (ms)

Comparing GPU and Host CPU results...
  GPU sum: 16777290.000000
  CPU sum: 16777294.395033
  Relative difference: 2.619631E-07 


```
