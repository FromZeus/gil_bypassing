import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
import os
import threading
import affinity
import cython

#os.system("taskset -p 0xff %d" % os.getpid())
affinity.set_process_affinity_mask(0, 2 ** mp.cpu_count() - 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long [:] primes(long kmax, long [:] res) nogil:
  cdef long n, k, i
  cdef long p[100000]
  cdef long [:] result = res

  if kmax > 100000:
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
      result[k] = n
    n = n + 1

  return result


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_multi(np.ndarray[np.int64_t, ndim=1] kmax):
  #pool = Pool()
  #for el in kmax:
  #  res.append(pool.apply_async(primes, (el,)).get())
  #pool = []
  #for el in kmax:
  #  t = threading.Thread(target=primes, args=(el,))
  #  t.start()
  #  pool.append(t)
  #for el in pool:
  #  el.join()
  cdef long n, i
  cdef long [:] m_kmax = kmax
  cdef long result[100000]
  cdef long [:] res = result
  n = len(kmax)
  for i in prange(n, nogil=True):
    primes(m_kmax[i], res)
  #return res