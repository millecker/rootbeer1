cl /I"C:\Program Files\Java\jdk1.6.0_45\include" /I"C:\Program Files\Java\jdk1.6.0_45\include\win32" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" org/trifort/rootbeer/runtime/FixedMemory.c /link "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64\cuda.lib" "C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib\x64\kernel32.lib" /DLL /OUT:rootbeer_x64.dll /MACHINE:X64

cl /I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include" -EHsc -c at/illecker/HadoopUtils.cpp
cl /I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include" -EHsc -c at/illecker/HostDeviceInterface.cpp
cl /I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include" -EHsc -c at/illecker/SocketClient.cpp
cl /I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include" -EHsc -c at/illecker/HostMonitor.cpp
cl /I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include"  /I"C:\Program Files\Java\jdk1.6.0_45\include" /I"C:\Program Files\Java\jdk1.6.0_45\include\win32" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" -EHsc -c at/illecker/HostMonitor.cpp org/trifort/rootbeer/runtime/HamaPeer.cpp

cl /I"C:\Program Files\Java\jdk1.6.0_45\include" /I"C:\Program Files\Java\jdk1.6.0_45\include\win32" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" -c org/trifort/rootbeer/runtime/CUDARuntime.c
cl /I"C:\Program Files\Java\jdk1.6.0_45\include" /I"C:\Program Files\Java\jdk1.6.0_45\include\win32" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" -c org/trifort/rootbeer/runtime/CUDAContext.c

lib /OUT:rootbeer_cuda_x64.dll /MACHINE:X64 HadoopUtils.obj SocketClient.obj HostDeviceInterface.obj HostMonitor.obj HamaPeer.obj CUDARuntime.obj CUDAContext.obj "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64\cuda.lib"