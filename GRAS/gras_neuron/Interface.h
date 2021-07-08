#ifndef GRAS_NEURON_INTERFACE_H
#define GRAS_NEURON_INTERFACE_H

#include <string>
#include <vector>
#include <iostream>

// global name of the models
const char GENERATOR = 'g';
const char INTER = 'i';
const char MOTO = 'm';
const char MUSCLE = 'u';
const char AFFERENTS = 'a';
const int neurons_in_group = 50;

enum string_code {air, toe, plt, quadru, normal, qpz, str, s6, s13, s21};

class Group {
public:
    Group() = default;
    std::string group_name;
    char model{};
    unsigned int id_start{};
    unsigned int id_end{};
    unsigned int group_size{};
};


namespace std{
    Group form_group(const string &group_name,
                     int nrns_in_group = neurons_in_group,
                     const char model = INTER,
                     const int segs = 1);

    void connect_fixed_indegree(Group &pre_neurons, Group &post_neurons, double delay, double weight, int indegree=50, short high_distr=0);

    void simulate(void (*network)());

    void custom(void (*t_network)(), int steps = 1, int test_ind=1, int TEST = 0, int E2F_coef = 1, int V0v2F_coef = 1, int QUADRU_Ia = 1,
                string_code mode = quadru, string_code pharma = normal, string_code speed = s13);
}


#endif //GRAS_NEURON_INTERFACE_H
