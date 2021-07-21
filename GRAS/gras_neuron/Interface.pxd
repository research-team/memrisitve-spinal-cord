# distutils: language = c++
from libcpp.string cimport string

# cdef extern from "structs.h":


cdef extern from "structs.h":

    cdef cppclass Group:
        Group()
        string group_name
        const char model
        unsigned int id_start
        unsigned int id_end
        unsigned int group_size
    cdef Group form_group(string &, int, char, int)
# cdef extern from "core.cu" namespace "std":
