# distutils: language = c++
# distutils: sources = core.cu

from Interface cimport *

# def py_form_group(name_group, nrns_in_grp=50, mode='i', seg=1):
#     cdef string name = bytes(name_group, 'utf-8')
#     cdef int nrns = nrns_in_grp
#     cdef char model = ord(mode)
#     cdef int segs = seg
#     form_group(name, nrns, model, segs)

cdef class Py_Group():
    cdef Group cpp_group
    def __init__(self, name):
        self.cpp_group = form_group(bytes(name, 'utf-8'), 50, 'i', 1)

def py_connect(pre_group : Py_Group, post_group: Py_Group, py_delay: float, py_weight: float, py_indegree=50, py_high=0):
    connect_fixed_indegree(pre_group.cpp_group,post_group.cpp_group, py_delay, py_weight, py_indegree, py_high)