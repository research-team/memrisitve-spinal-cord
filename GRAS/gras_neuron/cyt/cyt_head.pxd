cdef extern from "core.cu":
    pass

cdef extern from "struct.h":
    cdef enum string_code:
        air
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
        string group_name
        char model
        unsigned int id_start
        unsigned int id_end
        unsigned int group_size
