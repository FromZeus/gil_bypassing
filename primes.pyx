import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
cimport numpy as np
import os
import threading
import affinity

#os.system("taskset -p 0xff %d" % os.getpid())
affinity.set_process_affinity_mask(0,2**mp.cpu_count()-1)

def primes(int kmax):
  cdef int n, k, i
  cdef int p[100000]
  result = []
  if kmax > 1000:
     kmax = 100000
  k = 0
  n = 2
  while k < kmax:
    i = 0
    while i < k and n % p[i] != 0:
      i = i + 1
    if i == k:
      p[k] = n
      k = k + 1
      result.append(n)
    n = n + 1
  return result

def start_mult(np.ndarray[np.int64_t, ndim=1] kmax):
  #pool = Pool()
  #res = []
  #for el in kmax:
  #  res.append(pool.apply_async(primes, (el,)).get())
  pool = []
  for el in kmax:
    t = threading.Thread(target=primes, args=(el,))
    t.start()
    pool.append(t)
  for el in pool:
    el.join()
  #return res