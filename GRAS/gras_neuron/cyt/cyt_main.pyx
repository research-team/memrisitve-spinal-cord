# distutils: language = c++

from cyt_head cimport *

cdef extern from "core.cu":
    Group form_group(const string & group_name,
                     int nrns_in_group = neurons_in_group,
                     const char model = INTER,
                     const int segs = 1)

    void connect_fixed_indegree(Group & pre_neurons, Group & post_neurons, double delay, double weight, int indegree=50,
                                short high_distr=0)

    void simulate(int test_index, void network())

    void custom(void time_network(), int step_number = 1, int TEST = 0, int E2F_coef = 1, int V0v2F_coef = 1,
                int QUADRU_Ia = 1, string_code mode = quadru, string_code pharma = normal, string_code speed = s13)

cpdef call_form_group():
    cdef Group group
    form_group(group)
    return group

cpdef call_connect_fixed_indegree():
    cdef Group pre_group, post_group
    cdef double delay, weight
    connect_fixed_indegree(pre_group, post_group, delay, weight)

cpdef call_simulate():
    cdef int index
    cdef void func
    simulate(index, func)

cpdef call_custom():
    cdef void func
    custom(func)
