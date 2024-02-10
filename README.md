# brute-CUDA

brute - port for CUDA (works on Linux)

# known bugs
 - warnings + errors

Originally source code by Copyright (C) 2024 Smiril - sonar@gmx.com

Parts Copyright (C) by NVIDIA (see header)


## Compile

```shell
sudo make
```
## Usage
```shell
$ ./simpleMultiGPU hashes.txt
```

## Example

 **sha256**
```shell
$ ./simpleMultiGPU test.txt
Starting simpleMultiGPU
CUDA-capable device count: 1
Generating input data...

Computing with 1 GPUs...
  GPU Processing time: 11.510000 (ms)

Computing with Host CPU...

Comparing GPU and Host CPU results...
  GPU sum: 16777290.000000
  CPU sum: 16777295.395033
  Relative difference: 3.215675E-07 
```
