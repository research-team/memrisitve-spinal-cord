# distutils: language = c++
# distutils: sources = core.cu

from libcpp.string cimport string
from Interface cimport *

cpdef ff_form_group():
    cdef string name
    form_group(name, 50, "i", 1)
