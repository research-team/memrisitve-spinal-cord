from libcpp.string cimport string 

cdef extern from "Interface.h":

    cdef enum string_code:
        air = 0
        toe
        plt
        quadru
        normal
        qpz
        str
        s6
        s13
        s21

    cdef cppclass Group:
        Group()


cdef extern from "Interface.h" namespace "std":

    cdef void connect_fixed_indegree(Group & pre_neurons, Group & post_neurons, double delay, double weight, int indegree,
                                short high_distr)

    cdef void simulate(void (*network)())

    cdef void custom(void (*t_network)(), int steps, int test_ind, int TEST, int E2F_coef, int V0v2F_coef,
                int QUADRU_Ia,
                string_code mode, string_code pharma, string_code speed)

    cdef Group form_group(const string & group_name, int nrns_in_group, const char model, const int segs )
