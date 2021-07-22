#ifndef NEURON_GRAS_STRUCTS_H
#define NEURON_GRAS_STRUCTS_H

#include <string>
#include <vector>
#include <iostream>

// customize
enum string_code {air, toe, plt, quadru, normal, qpz, str, s6, s13, s21};

// global name of the models
const char GENERATOR = 'g';
const char INTER = 'i';
const char MOTO = 'm';
const char MUSCLE = 'u';
const char AFFERENTS = 'a';

 class Group {
    public:
        Group() = default;
        std::string group_name;
        char model{};
        unsigned int id_start{};
        unsigned int id_end{};
        unsigned int group_size{};
 };

void connect_fixed_indegree(Group &pre_neurons, Group &post_neurons, double delay, double weight, int indegree, short high_distr);
Group form_group(const std::string &group_name, int nrns_in_group, const char model, const int segs);



// struct for human-readable initialization of connectomes
struct GroupMetadata {
	Group group;
	double *g_exc;                // [nS] array of excitatory conductivity
	double *g_inh;                // [nS] array of inhibition conductivity
	double *voltage_array;        // [mV] array of membrane potential
	std::vector<double> spike_vector;  // [ms] spike times

	explicit GroupMetadata(Group group, int time) {
		this->group = std::move(group);
		voltage_array = new double[time];
		g_exc = new double[time];
		g_inh = new double[time];
	}
};

struct Generators {
	unsigned int *time_end;
	unsigned int *nrn_id;
	unsigned int *freq_in_steps;
	unsigned int *spike_each_step;
	unsigned int size;      // array size
};

// common neuron's parameters
// also from https://www.cell.com/neuron/pdfExtended/S0896-6273(16)00010-6
struct Parameters {
	unsigned int *nrn_start_seg;   // [index]
	char *models;           // [str] model's names
	double *Cm;             // [uF / cm2] membrane capacitance
	double *ref_t;             //
	double *gnabar;         // [S / cm2] the maximal fast Na+ conductance
	double *gkbar;          // [S / cm2] the maximal slow K+ conductance
	double *gl;             // [S / cm2] the maximal leak conductance
	double *Ra;             // [Ohm cm] axoplasmic resistivity
	double *diam;           // [um] soma compartment diameter
	double *length;         // [um] soma compartment length
	double *ena;            // [mV] Na+ reversal (equilibrium, Nernst) potential
	double *ek;             // [mV] K+ reversal (equilibrium, Nernst) potential
	double *el;             // [mV] Leakage reversal (equilibrium) potential
	// moto neuron's properties
	// https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=189786
	// https://journals.physiology.org/doi/pdf/10.1152/jn.2002.88.4.1592
	double *gkrect;         // [S / cm2] the maximal delayed rectifier K+ conductance
	double *gcaN;           // [S / cm2] the maximal N-type Ca2+ conductance
	double *gcaL;           // [S / cm2] the maximal L-type Ca2+ conductance
	double *gcak;           // [S / cm2] the maximal Ca2+ activated K+ conductance
	// synapses' parameters
	double *E_ex;           // [mV] excitatory reversal (equilibrium) potential
	double *E_inh;          // [mV] inhibitory reversal (equilibrium) potential
	double *tau_exc;        // [ms] rise time constant of excitatory synaptic conductance
	double *tau_inh1;       // [ms] rise time constant of inhibitory synaptic conductance
	double *tau_inh2;       // [ms] decay time constant of inhibitory synaptic conductance
	unsigned int size;      // array size
};

// neuron states
struct States {
	double *Vm;             // [mV] array for three compartments volatge
	double *n;              // [0..1] compartments channel, providing the kinetic pattern of the L conductance
	double *m;              // [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	double *h;              // [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	double *l;              // [0..1] inward rectifier potassium (Kir) channel
	double *s;              // [0..1] nodal slow potassium channel
	double *p;              // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	double *hc;             // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	double *mc;             // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	double *cai;            //
	double *I_Ca;           // [nA] Ca ionic currents
	double *NODE_A;         // the effect of this node on the parent node's equation
	double *NODE_B;         // the effect of the parent node on this node's equation
	double *NODE_D;         // diagonal element in node equation
	double *const_NODE_D;   // const diagonal element in node equation (performance)
	double *NODE_RHS;       // right hand side in node equation
	double *NODE_RINV;      // conductance uS from node to parent
	double *NODE_AREA;      // area of a node in um^2
	double *EXT_V;          // [mV] array for three compartments volatge
	double *EXT_A;          // the effect of this node on the parent node's equation
	double *EXT_B;          // the effect of the parent node on this node's equation
	double *EXT_D;          // diagonal element in node equation
	double *EXT_RHS;        // right hand side in node equation
	unsigned int size;      // array size
	unsigned int ext_size;  // array size for ext
};

struct Neurons {
	bool *has_spike;        // spike flag for each neuron
	bool *spike_on;         // special flag to prevent fake spike detecting
	double *g_exc;          // [S] excitatory conductivity level
	double *g_inh_A;        // [S] inhibitory conductivity level
	double *g_inh_B;        // [S] inhibitory conductivity level
	double *factor;         // [const]
	unsigned int *ref_time;     // [const]
	unsigned int *ref_time_timer;   // [const]
	unsigned int size;      // array size
};

struct Synapses {
	int *syn_pre_nrn;       // [id] list of pre neurons ids
	int *syn_post_nrn;      // [id] list of pre neurons ids
	double *syn_weight;     // [S] list of synaptic weights
	int *syn_delay;         // [ms * dt] list of synaptic delays in steps
	int *syn_delay_timer;   // [ms * dt] list of synaptic timers, shows how much left to send signal
	unsigned int size;      // array size
};





#endif //NEURON_GRAS_STRUCTS_H

////
