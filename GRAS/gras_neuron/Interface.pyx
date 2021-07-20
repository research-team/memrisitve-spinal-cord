# distutils: language = c++
# distutils: sources = core.cu

from libcpp.string cimport string
cimport Interface

cpdef ff_form_group(name):
    form_group(name, 50, "i", 1)
