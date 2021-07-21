# distutils: language = c++
# distutils: sources = core.cu

from Interface cimport *

def py_form_group(name_group, nrns_in_grp):
    cdef string name = bytes(name_group, 'utf-8')
    cdef int nrns = nrns_in_grp

    form_group(name, nrns, 'i', 1)
