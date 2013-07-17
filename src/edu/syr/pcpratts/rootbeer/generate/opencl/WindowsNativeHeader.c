#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>

DWORD threadIdxxKey;
DWORD blockIdxxKey;
DWORD blockDimxKey;
<<<<<<< HEAD
=======
DWORD gridDimxKey;
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7
CRITICAL_SECTION atom_add_mutex;
CRITICAL_SECTION thread_id_mutex;
CRITICAL_SECTION barrier_mutex;
CRITICAL_SECTION thread_gate_mutex;

void lock_atom_add(){
  EnterCriticalSection(&atom_add_mutex);
}

void unlock_atom_add(){
  LeaveCriticalSection(&atom_add_mutex);
}

void lock_thread_id(){
  EnterCriticalSection(&thread_id_mutex);
}

void unlock_thread_id(){
  LeaveCriticalSection(&thread_id_mutex);
}

void barrier_mutex_lock(){
  EnterCriticalSection(&barrier_mutex);
}

void barrier_mutex_unlock(){
  LeaveCriticalSection(&barrier_mutex);
}

void thread_gate_mutex_lock(){
  EnterCriticalSection(&thread_gate_mutex);
}

void thread_gate_mutex_unlock(){
  LeaveCriticalSection(&thread_gate_mutex);
}

int getThreadId(){
  return getBlockIdxx() * getBlockDimx() + getThreadIdxx();
}

int getThreadIdxx(){
  return (int) TlsGetValue(threadIdxxKey);
}

int getBlockIdxx(){
  return (int) TlsGetValue(blockIdxxKey);
}

int getBlockDimx(){
  return (int) TlsGetValue(blockDimxKey);
}

<<<<<<< HEAD
=======
int getGridDimx(){
  return (int) TlsGetValue(gridDimxKey);
}
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7
long long java_lang_System_nanoTime(char * gc_info, int * exception){
  SYSTEMTIME system_time;
  GetSystemTime(&system_time);
  return system_time.wMilliseconds;
}

void edu_syr_pcpratts_sleep(int micro_seconds){
  int milliseconds;
  milliseconds = micro_seconds / 1000;
  if(milliseconds < 0){
    milliseconds = 1;
  }
  Sleep(milliseconds);
}