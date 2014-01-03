## Rootbeer

The Rootbeer GPU Compiler makes it easy to use Graphics Processing Units from
within Java.

Rootbeer is more advanced than CUDA or OpenCL Java Language Bindings. With 
bindings the developer must serialize complex graphs of objects into arrays
of primitive types. With Rootbeer this is done automatically. Also with language
bindings, the developer must write the GPU kernel in CUDA or OpenCL. With
Rootbeer a static analysis of the Java Bytecode is done (using Soot) and CUDA
code is automatically generated.

See `doc/hpcc_rootbeer.pdf` for the conference slides from HPCC-2012.
See `doc/rootbeer1_paper.pdf` for the conference paper from HPCC-2012.

Rootbeer is licensed under the MIT license.

## Development notes

Rootbeer was created using Test Driven Development and testing is essentially
important in Rootbeer. Rootbeer is 20k lines of product code and 8.8k of test code
and 48/58 tests currently pass on Windows, Linux and Mac. The Rootbeer test case suite 
covers every aspect of the Java Programming language except:
  1. native methods
  2. reflection
  3. dynamic method invocation
  4. sleeping while inside a monitor. 
  
The original publication for Rootbeer was in HPCC-2012.  
  "Rootbeer: Seamlessly using GPUs from Java"  
  Philip C. Pratt-Szeliga, James W. Fawcett, Roy D. Welch.  
  HPCC-2012.  

This work is supported by the National Science Foundation grant number 
MCB-0746066 to R.D.W. and God is Most High.

Work is on-going to improve performance with Rootbeer. Currently Rootbeer has
competative speed when using single-dimensional arrays of primitive types.

## Building

1. Clone the github repo to `rootbeer1/`
2. `cd rootbeer1/`
3. `ant jar`
4. `./pack-rootbeer` (linux) or `./pack-rootbeer.bat` (windows)
5. Use the `rootbeer1/Rootbeer.jar` (not `dist/Rootbeer1.jar`)

## Pre-Built Binaries  

See here: http://rbcompiler.com/download.html

## Command Line Options

* -nemu = test without GPU
* -runeasytests = run test suite to see if things are working
* -printdeviceinfo = print out information regarding your GPU
* -maxrregcount = sent to CUDA compiler to limit register count
* -noarraychecks = remove array out of bounds checks once you get your application to work
* -nodoubles = you are telling rootbeer that there are no doubles and we can compile with older versions of CUDA
* -noexceptions = remove exception checking
* -shared-mem-size = specify the shared memory size
* -32bit = compile with 32bit
* -64bit = compile with 64bit (if you are on a 64bit machine you will want to use just this)

Once you get started, you will find you want to use a combination of -maxregcount, -shared-mem-size and the thread count sent to the GPU to control occupancy.

## CUDA Setup

You need to have the CUDA Toolkit and CUDA Driver installed to use Rootbeer.
Download it from http://www.nvidia.com/content/cuda/cuda-downloads.html

## API

See the following links for help on the API:  
1. http://rbcompiler.com/  
2. http://rbcompiler.com/features.html  
3. https://github.com/pcpratts/rootbeer1/tree/develop/gtc2013/Matrix  

## About

Rootbeer is written by:

Phil Pratt-Szeliga  
Syracuse University  
pcpratts@trifort.org  
http://trifort.org/  
http://rbcompiler.com/
