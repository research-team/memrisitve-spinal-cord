# distutils: language = c++
from libcpp.string cimport string

cdef extern from "structs.h":
    cdef cppclass Group:
        Group()
        string group_name;
        char model
        unsigned int id_start
        unsigned int id_end
        unsigned int group_size

cdef extern from "core.cu":
    Group form_group(const string & group_name,
                     int nrns_in_group,
                     char model,
                     int segs)
