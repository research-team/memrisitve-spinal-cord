# distutils: language = c++
from libcpp.string cimport string

cdef extern from "structs.h":
    cppclass Group:
        Group()
        string group_name;
        char model
        unsigned int id_start
        unsigned int id_end
        unsigned int group_size

cdef extern from "structs.h":
    Group form_group(string &, int, char,int)
