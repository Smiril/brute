# brute 

brute - port for Mac OSX (works on Mac OSX)

# known bugs
 - warnings + errors
   - Linking errors
   - Segfaults

Originally source code by Copyright (C) 2024 Smiril - sonar@gmx.com


## Compile

```shell
sudo make
```
## Install

```shell
sudo make install
```

## uninstall

```shell
sudo make uninstall
```

## Usage
```shell
$ ./brute
shaCrack! 0.1 by Smiril (sonar@gmx.com)

USAGE: brute [--threads NUM] --hash [hashes.ext]
       For more information please run "brute --help"
```

## Example

 **sha256**
```shell
$ ./brute --threads 2 --hash test.txt
shaCrack! 0.1 by Smiril (sonar@gmx.com)

Probing: 'test' [1 pwds/sec]
Probing: '123' [1 pwds/sec]
Probing: 'password' [1 pwds/sec]
GOOD: password 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8 cracked: 'password'
```
