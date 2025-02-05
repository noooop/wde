import cython
import numpy as np

cimport numpy as cnp
from libc.string cimport memcpy

cnp.import_array()

@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
cdef void c_cython_memcpy(short * from_kv_cache_ptr,
                     short * to_kv_cache_ptr,
                     int[:] from_ids,
                     int[:] to_ids,
                     int n,
                     int count):

    cdef int f, t, i

    for i in range(n):
        f = from_ids[i]
        t = to_ids[i]
        memcpy(to_kv_cache_ptr+t*count, from_kv_cache_ptr+f*count, count * 2)


@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cython_memcpy(from_kv_cache, to_kv_cache, tasks):
    cdef int m, _, n, i, b, count

    m, _, n = tasks.shape
    b = from_kv_cache.shape[0]
    b, count = from_kv_cache.reshape(b, -1).shape

    from_kv_cache = from_kv_cache.reshape(-1)
    to_kv_cache = to_kv_cache.reshape(-1)

    tasks_view = cython.declare(cython.int[:, :, :], tasks)
    from_kv_cache_view = cython.declare(cython.short[:], from_kv_cache)
    to_kv_cache_view = cython.declare(cython.short[:], to_kv_cache)

    from_kv_cache_ptr = & from_kv_cache_view[0]
    to_kv_cache_ptr = & to_kv_cache_view[0]

    for i in range(m):
        c_cython_memcpy(from_kv_cache_ptr, to_kv_cache_ptr, tasks_view[i][0], tasks_view[i][1], n, count)