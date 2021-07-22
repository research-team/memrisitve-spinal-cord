# distutils: language = c++
from libcpp.string cimport string

# cdef extern from "structs.h":


cdef extern from "structs.h":

    cdef cppclass Group:
        Group() except +
        string group_name
        const char model
        unsigned int id_start
        unsigned int id_end
        unsigned int group_size

    cdef Group form_group(string &, int, char, int)

    cdef void connect_fixed_indegree(Group &pre_neurons, Group &post_neurons, double delay, double weight,
                                     int indegree, short high_distr)

