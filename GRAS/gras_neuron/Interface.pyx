# distutils: language = c++

from Interface cimport *

cdef class PyGroup():
    cdef Group c_group
    def __cinit__(self):
        self.c_group = Group()

def call_connect_fixed_indegree():
    cdef Group pre_group, post_group
    cdef double delay, weight
    connect_fixed_indegree(pre_group, post_group, delay, weight, 50, 0)

def call_simulate():
    cdef void (*func)()
    simulate(func)

def call_custom():
    cdef void (*func)()
    custom(func, 1, 1, 0, 1, 1, 1, quadru, normal, s13)


def call_form_group():
    cdef const string name
    form_group(name, 50, 'i', 1)