# distutils: language = c++
# distutils: sources = core.cpp

import cytepes
from cyt_head cimport *


cpdef call_connect_fixed_indegree():
    cdef Group pre_group, post_group
    cdef double delay, weight
    connect_fixed_indegree(pre_group, post_group, delay, weight, 50, 0)

cpdef call_simulate():
    cdef void (*func)()
    simulate(func)

cpdef call_custom():
    cdef void (*func)()
    custom(func, 1, 1, 0, 1, 1, 1, quadru, normal, s13)


cpdef call_form_group():
    cdef const string name
    form_group(name, 50, 'i', 1)
