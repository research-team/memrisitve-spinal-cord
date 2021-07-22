from Interface import Py_Group, py_connect

x = Py_Group('a')
a = Py_Group('b')
py_connect(x, a, 1, 0.025)