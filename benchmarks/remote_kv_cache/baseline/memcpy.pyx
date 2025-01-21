import cython
import numpy as np

cimport numpy as cnp
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

cnp.import_array()

@cython.cdivision
@cython.boundscheck(False)
@cython.nogil
cdef void c_cython_memcpy(uintptr_t from_kv_cache,
                     uintptr_t to_kv_cache,
                     int[:] from_ids,
                     int[:] to_ids,
                     int n,
                     int count):
    cdef short * from_kv_cache_ptr = < short * > from_kv_cache
    cdef short * to_kv_cache_ptr = < short * > to_kv_cache

    cdef int f, t, i

    for i in range(n):
        f = from_ids[i]
        t = to_ids[i]
        memcpy(to_kv_cache_ptr+t*count, from_kv_cache_ptr+f*count, count * 2)


cpdef cython_memcpy(from_kv_cache, to_kv_cache, tasks):
    from_kv_cache_ptr = from_kv_cache.data_ptr()
    to_kv_cache_ptr = to_kv_cache.data_ptr()
    count = from_kv_cache[0].nelement()

    m = len(tasks)
    n = len(tasks[0][0])

    tasks = np.array(tasks, dtype=np.int32)
    tasks_view = cython.declare(cython.int[:, :, :], tasks)

    for i in range(m):
        c_cython_memcpy(from_kv_cache_ptr, to_kv_cache_ptr, tasks_view[i][0], tasks_view[i][1], n, count)