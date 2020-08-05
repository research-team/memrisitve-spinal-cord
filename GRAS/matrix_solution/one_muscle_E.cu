#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <ctime>
#include <stdexcept>
#include <random>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include "Group.cpp"

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#define __shared__
#endif

using namespace std;

const unsigned int syn_outdegree = 27;
const unsigned int neurons_in_ip = 196;
const unsigned int neurons_in_moto = 169;
const unsigned int neurons_in_group = 20;
const unsigned int neurons_in_afferent = 196;

const int SPEED = 25;
const int EES_FREQ = 40;
const float INH_COEF = 1.0f;

// 6 CMS = 125	[ms]
// 15 CMS = 50	[ms]
// 21 CMS = 25	[ms]

// stuff variable
unsigned int global_id = 0;
const float T_SIMULATION = 125;
const float SIM_STEP = 0.25;
const unsigned int sim_time_in_steps = (unsigned int)(T_SIMULATION / sim_step);

__host__
int ms_to_step(float ms) { return (int)(ms / sim_step); }

struct SynapseMetadata{
	// struct for human-readable initialization of connectomes
	int post_id;
	int synapse_delay;
	float synapse_weight;

	SynapseMetadata() = default;
	SynapseMetadata(int post_id, float synapse_delay, float synapse_weight){
		this->post_id = post_id;
		this->synapse_delay = static_cast<int>(synapse_delay * (1 / sim_step) + 0.5); // round
		this->synapse_weight = synapse_weight;
	}
};

Group form_group(string group_name, int nrns_in_group = neurons_in_group) {
	// form structs of neurons global ID and groups name
	Group group = Group();

	group.group_name = group_name;
	group.id_start = global_id;
	group.id_end = global_id + nrns_in_group - 1;
	group.group_size = nrns_in_group;

	global_id += nrns_in_group;

	printf("Formed %s IDs [%d ... %d] = %d\n",
		   group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

// Form neuron groups
Group CV1 = form_group("CV1");
Group CV2 = form_group("CV2");

Group CV3 = form_group("CV3");
Group CV4 = form_group("CV4");
Group CV5 = form_group("CV5");

Group D1_1 = form_group("D1_1");
Group D1_2 = form_group("D1_2");
Group D1_3 = form_group("D1_3");
Group D1_4 = form_group("D1_4");

Group D2_1 = form_group("D2_1");
Group D2_2 = form_group("D2_2");
Group D2_3 = form_group("D2_3");
Group D2_4 = form_group("D2_4");

Group D3_1 = form_group("D3_1");
Group D3_2 = form_group("D3_2");
Group D3_3 = form_group("D3_3");
Group D3_4 = form_group("D3_4");

Group D4_1 = form_group("D4_1");
Group D4_2 = form_group("D4_2");
Group D4_3 = form_group("D4_3");
Group D4_4 = form_group("D4_4");

Group D5_1 = form_group("D5_1");
Group D5_2 = form_group("D5_2");
Group D5_3 = form_group("D5_3");
Group D5_4 = form_group("D5_4");

Group G1_1 = form_group("G1_1");
Group G1_2 = form_group("G1_2");
Group G1_3 = form_group("G1_3");

Group G2_1 = form_group("G2_1");
Group G2_2 = form_group("G2_2");
Group G2_3 = form_group("G2_3");

Group G3_1 = form_group("G3_1");
Group G3_2 = form_group("G3_2");
Group G3_3 = form_group("G3_3");

Group G4_1 = form_group("G4_1");
Group G4_2 = form_group("G4_2");
Group G4_3 = form_group("G4_3");

Group G5_1 = form_group("G5_1");
Group G5_2 = form_group("G5_2");
Group G5_3 = form_group("G5_3");

Group IP_E = form_group("IP_E", neurons_in_ip);
Group IP_F = form_group("IP_F", neurons_in_ip);

Group MP_E = form_group("MP_E", neurons_in_moto);
Group MP_F = form_group("MP_F", neurons_in_moto);

Group EES = form_group("EES");
Group Ia = form_group("Ia", neurons_in_afferent);

Group I3 = form_group("inh_group3");
Group I4 = form_group("inh_group4");
Group I5 = form_group("inh_group5");

Group E1 = form_group("ees_group1");
Group E2 = form_group("ees_group2");
Group E3 = form_group("ees_group3");
Group E4 = form_group("ees_group4");

Group R_E = form_group("R_E");
Group R_F = form_group("R_F");

Group Ia_E = form_group("Ia_E");
Group Ia_F = form_group("Ia_F");
Group Ib_E = form_group("Ib_E");
Group Ib_F = form_group("Ib_F");

// Global vectors of SynapseMetadata of synapses for each neuron
vector<vector<SynapseMetadata>> metadatas(global_id, vector<SynapseMetadata>());

// Global stuff variables
bool* has_multimeter;
bool* has_generator;
int* begin_spiking;
int* end_spiking;
int* spiking_per_step;

// Parameters (const)
const float C = 100;        // [pF] membrane capacitance
const float V_rest = -72;   // [mV] resting membrane potential
const float V_thld = -55;   // [mV] spike threshold
const float k = 0.7;          // [pA * mV-1] constant ("1/R")
const float a = 0.02;         // [ms-1] time scale of the recovery variable U_m. Higher a, the quicker recovery
const float b = 0.2;          // [pA * mV-1] sensitivity of U_m to the sub-threshold fluctuations of the V_m
const float c = -80;        // [mV] after-spike reset value of V_m
const float d = 6;          // [pA] after-spike reset value of U_m
const float V_peak = 35;    // [mV] spike cutoff value

const float V_half = -65;      // [mV] is the half-activation voltage where 50% of the maximal conductance level is reached
const float k_slope = 1;       // defines the slope of the output function

__global__
void sim_kernel(float* old_v,
                float* old_u,
                float* nrn_current,
                int* nrn_ref_time,
                int* nrn_ref_time_timer,
                int* synapses_number,
                bool* has_spike,
                int** synapses_post_nrn_id,
                int** synapses_delay,
                int** synapses_delay_timer,
                float** synapses_weight,
                unsigned int nrn_size,
                bool* has_generator,
                bool* has_multimeter,
                float* multimeter_result,
                int* begin_spiking,
                int* end_spiking,
                int* spiking_per_step,
                // ToDo remove after debugging
                float* voltage_recording,
                float* current_recording,
                int* spike_recording) {


	// FixMe: hidden bug, but will work perfect if number of spikes will be lower than sim_step_time / 2 (usually)
	// FixMe: explanation -- each thread has local variable, but here is stride loop. So one thread do at least 2 job
	int local_spike_array_iter = 0;

	// get id of the thread
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	// the main simulation loop
	for (int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++) {
		// wait all threads
		__syncthreads();

		// neuron (tid = neuron id) stride loop (0, 1024, 1, 1025 ...)
		for (int tid = thread_id; tid < nrn_size; tid += blockDim.x * gridDim.x) {
			// spike-generator invoking
			if (has_generator[tid] &&
				sim_iter >= begin_spiking[tid] &&
				sim_iter < end_spiking[tid] &&
				(sim_iter % spiking_per_step[tid] == 0)) {
				nrn_current[tid] = 5000; // enough for spike
			}

			// todo check with the real neurobiology mechanism
			// absolute refractory period : calculate V_m and U_m WITHOUT synaptic weight (nrn_current)
			// action potential : calculate V_m and U_m WITH synaptic weight (nrn_current)
			if (nrn_ref_time_timer[tid] > 0)
				nrn_current[tid] = 0;

			float V_old = old_v[tid];
			float U_old = old_u[tid];
			float I_current = nrn_current[tid];

			// ToDo check this with biological data (aprx)
			// the maximal value of input current (10 000 pA = 10 nA)
			if (I_current > 10000)
				I_current = 10000;
			if (I_current < -10000)
				I_current = -10000;

			// re-calculate V_m and U_m
			float V_m = V_old + sim_step * (k * (V_old - V_rest) * (V_old - V_thld) - U_old + I_current) / C;
			float U_m = U_old + sim_step * a * (b * (V_old - V_rest) - U_old);

			// set bottom border of the membrane potential
			if (V_m < c)
				V_m = c;
			// set top border of the membrane potential
			if (V_m >= V_thld)
				V_m = V_peak;

			if (tid >= 1550 &&
				tid <= 1745 &&
				(sim_iter % Ia_afferent_spiking_per_step == 0)) {
				nrn_current[tid] = 5000;
			}

			// ToDo remove after debugging
			int index = sim_iter + tid * sim_time_in_steps;
			voltage_recording[index] = V_m;
			current_recording[index] = I_current;

			// threshold crossing (spike)
			if (V_m >= V_thld) {
				// set spike status
				has_spike[tid] = true;
				// redefine V_old and U_old
				old_v[tid] = c;
				old_u[tid] += d;
				// set the refractory period
				nrn_ref_time_timer[tid] = nrn_ref_time[tid];

				// ToDo remove after debugging
				spike_recording[local_spike_array_iter + tid * sim_time_in_steps] = sim_iter;
				local_spike_array_iter++;
			} else {
				// redefine V_old and U_old
				old_v[tid] = V_m;
				old_u[tid] = U_m;
			}

			// pointers to current neuronID synapses_delay_timer (decrease array calls)
			int *ptr_delay_timers = synapses_delay_timer[tid];
			// synapse updating loop
			for (int syn_id = 0; syn_id < synapses_number[tid]; syn_id++) {
				// add synaptic delay if neuron has spike
				if (has_spike[tid] && ptr_delay_timers[syn_id] == -1) {
					ptr_delay_timers[syn_id] = synapses_delay[tid][syn_id];
				}
				// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
				if (ptr_delay_timers[syn_id] == 0) {
					// post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
					atomicAdd(&nrn_current[ synapses_post_nrn_id[tid][syn_id] ], synapses_weight[tid][syn_id]);
					// make synapse timer a "free" for next spikes
					ptr_delay_timers[syn_id] = -1;
				}
				// update synapse delay timer
				if (ptr_delay_timers[syn_id] > 0) {
					ptr_delay_timers[syn_id]--;
				}
			} // end synapse updating loop

			// reset spike flag of the current neuron
			has_spike[tid] = false;

			// update currents of the neuron
			if (I_current != 0) {
				// decrease current potential for positive and negative current
				if (I_current > 0) nrn_current[tid] = I_current / 2;
				if (I_current < 0) nrn_current[tid] = I_current / 1.1f;
				// avoid the near value to 0
				if (I_current > 0 && I_current <= 1) nrn_current[tid] = 0;
				if (I_current <= 0 && I_current >= -1) nrn_current[tid] = 0;
			}

			// update the refractory period timer
			if (nrn_ref_time_timer[tid] > 0)
				nrn_ref_time_timer[tid]--;
		} // end of neuron stride loop

		// wait all threads
		__syncthreads();

	} // end of sim iteration
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons,
							 float syn_delay, float weight, int outdegree = syn_outdegree) {
	// connect neurons with uniform distribution and normal distributon for syn delay and weight
	weight *= (100 * 0.7);

	random_device rd;
	mt19937 gen(rd());	// Initialize pseudo-random number generator

	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	normal_distribution<float> weight_distr(weight, 2);
	normal_distribution<float> delay_distr(syn_delay, 0.1);

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = id_distr(gen);
			float syn_delay_dist = syn_delay;   // ToDo replace after tuning : delay_distr(gen);
			float syn_weight_dist = weight;     // ToDo replace after tuning : weight_distr(gen);
			metadatas.at(pre_id).push_back(SynapseMetadata(rand_post_id, syn_delay_dist, syn_weight_dist));
		}
	}

	printf("Connect %s with %s (1:%d). W=%.2f, D=%.1f\n",
		   pre_neurons.group_name.c_str(),
		   post_neurons.group_name.c_str(),
		   post_neurons.group_size,
		   weight,
		   syn_delay);
}

void group_add_multimeter(Group &nrn_group) {
	// function for adding multimeter to the neuron group
	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; nrn_id++) {
		has_multimeter[nrn_id] = true;
	}
	printf("Added multmeter to %s \n", nrn_group.group_name.c_str());
}

void group_add_spike_generator(Group &nrn_group, float start, float end, int hz){
	// function for adding spike-generator to the neuron group
	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; nrn_id++) {
		has_generator[nrn_id] = true;
		begin_spiking[nrn_id] = ms_to_step(start + 0.2);
		end_spiking[nrn_id] = ms_to_step(end);
		spiking_per_step[nrn_id] = ms_to_step(1000.0f / hz);
	}
	printf("Added generator to %s \n", nrn_group.group_name.c_str());
}

void init_extensor() {
	group_add_spike_generator(CV1, 0, speed, 200);
	group_add_spike_generator(CV2, speed, 2*speed, 200);
	group_add_spike_generator(CV3, 2*speed, 3*speed, 200);
	group_add_spike_generator(CV4, 3*speed, 5*speed, 200);
	group_add_spike_generator(CV5, 5*speed, 6*speed, 200);
	group_add_spike_generator(EES, 0, T_SIMULATION, EES_FREQ);

	connect_fixed_outdegree(CV3, inh_group3, 0.5, 15.0);
	connect_fixed_outdegree(CV4, inh_group4, 0.5, 15.0);
	connect_fixed_outdegree(CV5, inh_group5, 0.5, 15.0);

	connect_fixed_outdegree(inh_group3, G1_3, 2.8, 20.0);

	connect_fixed_outdegree(inh_group4, G1_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group4, G2_3, 1.0, 20.0);

	connect_fixed_outdegree(inh_group5, G1_3, 2.0, 20.0);
	connect_fixed_outdegree(inh_group5, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G4_3, 1.0, 20.0);

	/// D1
	// input from sensory
	connect_fixed_outdegree(CV1, D1_1, 1, 0.4);
	connect_fixed_outdegree(CV1, D1_4, 1, 0.4);
	connect_fixed_outdegree(CV2, D1_1, 1, 0.4);
	connect_fixed_outdegree(CV2, D1_4, 1, 0.4);
	// input from EES
	connect_fixed_outdegree(EES, D1_1, 2, 10); // ST value (?)
	connect_fixed_outdegree(EES, D1_4, 2, 10); // ST value (?)
	// inner connectomes
	connect_fixed_outdegree(D1_1, D1_2, 1, 1.0);
	connect_fixed_outdegree(D1_1, D1_3, 1, 10.0);
	connect_fixed_outdegree(D1_2, D1_1, 1, 7.0);
	connect_fixed_outdegree(D1_2, D1_3, 1, 10.0);
	connect_fixed_outdegree(D1_3, D1_1, 1, -10 * INH_COEF);
	connect_fixed_outdegree(D1_3, D1_2, 1, -10 * INH_COEF);
	connect_fixed_outdegree(D1_4, D1_3, 3, -20 * INH_COEF);
	// output to
	connect_fixed_outdegree(D1_3, G1_1, 3, 8);
	connect_fixed_outdegree(D1_3, ees_group1, 1.0, 60);

	// EES group connectomes
	connect_fixed_outdegree(ees_group1, ees_group2, 1.0, 20.0);

	/// D2
	// input from Sensory
	connect_fixed_outdegree(CV2, D2_1, 1, 0.8);
	connect_fixed_outdegree(CV2, D2_4, 1, 0.8);
	connect_fixed_outdegree(CV3, D2_1, 1, 0.8);
	connect_fixed_outdegree(CV3, D2_4, 1, 0.8);
	// input from Group (1)
	connect_fixed_outdegree(ees_group1, D2_1, 1.7, 0.8);
	connect_fixed_outdegree(ees_group1, D2_4, 1.7, 1.0);
	// inner connectomes
	connect_fixed_outdegree(D2_1, D2_2, 1.0, 3.0);
	connect_fixed_outdegree(D2_1, D2_3, 1.0, 10.0);
	connect_fixed_outdegree(D2_2, D2_1, 1.0, 7.0);
	connect_fixed_outdegree(D2_2, D2_3, 1.0, 20.0);
	connect_fixed_outdegree(D2_3, D2_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D2_3, D2_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D2_4, D2_3, 2.0, -20 * INH_COEF);
	// output to generator
	connect_fixed_outdegree(D2_3, G2_1, 1.0, 8);

	// EES group connectomes
	connect_fixed_outdegree(ees_group2, ees_group3, 1.0, 20.0);

	/// D3
	// input from Sensory
	connect_fixed_outdegree(CV3, D3_1, 1, 0.5);
	connect_fixed_outdegree(CV3, D3_4, 1, 0.5);
	connect_fixed_outdegree(CV4, D3_1, 1, 0.5);
	connect_fixed_outdegree(CV4, D3_4, 1, 0.5);
	// input from Group (2)
	connect_fixed_outdegree(ees_group2, D3_1, 1, 1.2);
	connect_fixed_outdegree(ees_group2, D3_4, 1, 1.2);
	// inner connectomes
	connect_fixed_outdegree(D3_1, D3_2, 1.0, 3.0);
	connect_fixed_outdegree(D3_1, D3_3, 1.0, 10.0);
	connect_fixed_outdegree(D3_2, D3_1, 1.0, 7.0);
	connect_fixed_outdegree(D3_2, D3_3, 1.0, 20.0);
	connect_fixed_outdegree(D3_3, D3_1, 1.0, -10 * INH_COEF);
	connect_fixed_outdegree(D3_3, D3_2, 1.0, -10 * INH_COEF);
	connect_fixed_outdegree(D3_4, D3_3, 2.0, -10 * INH_COEF);
	// output to generator
	connect_fixed_outdegree(D3_3, G3_1, 1, 25.0);
	// suppression of the generator
	connect_fixed_outdegree(D3_3, G1_3, 1.5, 30.0);

	// EES group connectomes
	connect_fixed_outdegree(ees_group3, ees_group4, 2.0, 20.0);

	/// D4
	// input from Sensory
	connect_fixed_outdegree(CV4, D4_1, 1, 0.5);
	connect_fixed_outdegree(CV4, D4_4, 1, 0.5);
	connect_fixed_outdegree(CV5, D4_1, 1, 0.5);
	connect_fixed_outdegree(CV5, D4_4, 1, 0.5);
	// input from Group (3)
	connect_fixed_outdegree(ees_group3, D4_1, 1, 1.2);
	connect_fixed_outdegree(ees_group3, D4_4, 1, 1.2);
	// inner connectomes
	connect_fixed_outdegree(D4_1, D4_2, 1.0, 3.0);
	connect_fixed_outdegree(D4_1, D4_3, 1.0, 10.0);
	connect_fixed_outdegree(D4_2, D4_1, 1.0, 7.0);
	connect_fixed_outdegree(D4_2, D4_3, 1.0, 20.0);
	connect_fixed_outdegree(D4_3, D4_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D4_3, D4_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D4_4, D4_3, 2.0, -20 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D4_3, G4_1, 3.0, 20.0);
	// suppression of the generator
	connect_fixed_outdegree(D4_3, G2_3, 1.0, 30.0);

	/// D5
	// input from Sensory
	connect_fixed_outdegree(CV5, D5_1, 1, 0.5);
	connect_fixed_outdegree(CV5, D5_4, 1, 0.5);
	// input from Group (4)
	connect_fixed_outdegree(ees_group4, D5_1, 1.0, 1.1);
	connect_fixed_outdegree(ees_group4, D5_4, 1.0, 1.0);
	// inner connectomes
	connect_fixed_outdegree(D5_1, D5_2, 1.0, 3.0);
	connect_fixed_outdegree(D5_1, D5_3, 1.0, 15.0);
	connect_fixed_outdegree(D5_2, D5_1, 1.0, 7.0);
	connect_fixed_outdegree(D5_2, D5_3, 1.0, 20.0);
	connect_fixed_outdegree(D5_3, D5_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D5_3, D5_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D5_4, D5_3, 2.5, -20 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D5_3, G5_1, 3, 8.0);
	// suppression of the genearator
	connect_fixed_outdegree(D5_3, G1_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G2_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G3_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G4_3, 1.0, 30.0);

	/// G1
	// inner connectomes
	connect_fixed_outdegree(G1_1, G1_2, 1.0, 10.0);
	connect_fixed_outdegree(G1_1, G1_3, 1.0, 15.0);
	connect_fixed_outdegree(G1_2, G1_1, 1.0, 10.0);
	connect_fixed_outdegree(G1_2, G1_3, 1.0, 15.0);
	connect_fixed_outdegree(G1_3, G1_1, 0.7, -70 * INH_COEF);
	connect_fixed_outdegree(G1_3, G1_2, 0.7, -70 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G1_1, IP_E, 3, 25.0);
	connect_fixed_outdegree(G1_1, IP_E, 3, 25.0);

	/// G2
	// inner connectomes
	connect_fixed_outdegree(G2_1, G2_2, 1.0, 10.0);
	connect_fixed_outdegree(G2_1, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(G2_2, G2_1, 1.0, 10.0);
	connect_fixed_outdegree(G2_2, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(G2_3, G2_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G2_3, G2_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G2_1, IP_E, 1.0, 65.0);
	connect_fixed_outdegree(G2_2, IP_E, 1.0, 65.0);

	/// G3
	// inner connectomes
	connect_fixed_outdegree(G3_1, G3_2, 1.0, 14.0);
	connect_fixed_outdegree(G3_1, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(G3_2, G3_1, 1.0, 12.0);
	connect_fixed_outdegree(G3_2, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(G3_3, G3_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G3_3, G3_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G3_1, IP_E, 2, 25.0);
	connect_fixed_outdegree(G3_1, IP_E, 2, 25.0);

	/// G4
	// inner connectomes
	connect_fixed_outdegree(G4_1, G4_2, 1.0, 10.0);
	connect_fixed_outdegree(G4_1, G4_3, 1.0, 10.0);
	connect_fixed_outdegree(G4_2, G4_1, 1.0, 5.0);
	connect_fixed_outdegree(G4_2, G4_3, 1.0, 10.0);
	connect_fixed_outdegree(G4_3, G4_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G4_3, G4_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G4_1, IP_E, 1.0, 17.0);
	connect_fixed_outdegree(G4_1, IP_E, 1.0, 17.0);

	/// G5
	// inner connectomes
	connect_fixed_outdegree(G5_1, G5_2, 1.0, 7.0);
	connect_fixed_outdegree(G5_1, G5_3, 1.0, 10.0);
	connect_fixed_outdegree(G5_2, G5_1, 1.0, 7.0);
	connect_fixed_outdegree(G5_2, G5_3, 1.0, 10.0);
	connect_fixed_outdegree(G5_3, G5_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G5_3, G5_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G5_1, IP_E, 2, 20.0);
	connect_fixed_outdegree(G5_1, IP_E, 2, 20.0);

	connect_fixed_outdegree(IP_E, MP_E, 1, 11);
	connect_fixed_outdegree(EES, MP_E, 2, 50);
	connect_fixed_outdegree(Ia, MP_E, 1, 1);
}

void save_result(int test_index,
				 float* voltage_recording,
				 float* current_recording,
				 int* spike_recording,
				 int neurons_number) {
	// save results for each neuron (voltage/current/spikes)
	char cwd[256];
	ofstream myfile;

	getcwd(cwd, sizeof(cwd));
	printf("[Test #%d] Save results to: %s \n", test_index, cwd);

	string new_name = "/volt.dat";
	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++)
			myfile << voltage_recording[sim_iter + nrn_id * sim_time_in_steps] << " ";
		myfile << "\n";
	}

	myfile.close();

	new_name = "/curr.dat";
	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++)
			myfile << current_recording[sim_iter + nrn_id * sim_time_in_steps] << " ";
		myfile << "\n";
	}

	myfile.close();

	new_name = "/spikes.dat";
	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++) {
			float spike_time = spike_recording[sim_iter + nrn_id * sim_time_in_steps] * sim_step;
			if (spike_time != 0)
				myfile << spike_time << " ";
		}
		myfile << "\n";
	}

	myfile.close();
}

template <typename type>
void memcpyHtD(type* gpu, type* host, int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

template <typename type>
void memcpyDtH(type* host, type* gpu, int size) {
	cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

template <typename type>
unsigned int datasize(int size) {
	return sizeof(type) * size;
}

template <typename type>
void init_array(type *array, int size, type value){
	for(int i = 0; i < size; i++)
		array[i] = value;
}

__host__
void simulate(int test_index) {
	int neurons_number = static_cast<int>(metadatas.size());

	float* gpu_old_v;
	float* gpu_old_u;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_timer;
	bool* gpu_has_spike;
	bool* gpu_has_generator;
	bool* gpu_has_multimeter;
	float* gpu_nrn_current;
	int* gpu_synapses_number;
	int* gpu_begin_spiking;
	int* gpu_end_spiking;
	int* gpu_spiking_per_step;
	float* gpu_multimeter_result;

	// ToDo remove after debugging
	float* gpu_voltage_recording;
	float* gpu_current_recording;
	int* gpu_spike_recording;

	int synapses_number[neurons_number];

	float old_v[neurons_number];
	init_array<float>(old_v, neurons_number, V_rest);

	float old_u[neurons_number];
	init_array<float>(old_u, neurons_number, 0);

	int nrn_ref_time[neurons_number];
	init_array<int>(nrn_ref_time, neurons_number, ms_to_step(3.0));

	int nrn_ref_timer[neurons_number];
	init_array<int>(nrn_ref_timer, neurons_number, -1);

	bool has_spike[neurons_number];
	init_array<bool>(has_spike, neurons_number, false);

	float nrn_current[neurons_number];
	init_array<float>(nrn_current, neurons_number, 0);

	float multimeter_result[sim_time_in_steps];
	init_array<float>(multimeter_result, sim_time_in_steps, 0);

	// ToDo remove after debugging
	float* voltage_recording = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	init_array<float>(voltage_recording, neurons_number * sim_time_in_steps, 0);
	float* current_recording = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	init_array<float>(current_recording, neurons_number * sim_time_in_steps, 0);
	int* spike_recording = (int *)malloc(datasize<int *>(neurons_number * sim_time_in_steps));
	init_array<int>(spike_recording, neurons_number * sim_time_in_steps, 0);

	has_multimeter = (bool *)malloc(datasize<bool *>(neurons_number));
	has_generator = (bool *)malloc(datasize<bool *>(neurons_number));
	begin_spiking = (int *)malloc(datasize<int *>(neurons_number));
	end_spiking = (int *)malloc(datasize<int *>(neurons_number));
	spiking_per_step = (int *)malloc(datasize<int *>(neurons_number));

	// init connectomes
	init_extensor();

	int **gpu_synapses_post_nrn_id, **synapses_post_nrn_id = (int **)malloc(datasize<int* >(neurons_number));
	int **gpu_synapses_delay, **synapses_delay = (int **)malloc(datasize<int* >(neurons_number));
	int **gpu_synapses_delay_timer, **synapses_delay_timer = (int **)malloc(datasize<int* >(neurons_number));
	float **gpu_synapses_weight, **synapses_weight = (float **)malloc(datasize<float* >(neurons_number));

	// fill arrays of synapses
	for(int neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		int syn_count = static_cast<int>(metadatas.at(neuron_id).size());

		int tmp_synapses_post_nrn_id[syn_count];
		int tmp_synapses_delay[syn_count];
		int tmp_synapses_delay_timer[syn_count];
		float tmp_synapses_weight[syn_count];

		int syn_id = 0;
		for(SynapseMetadata metadata : metadatas.at(neuron_id)) {
			tmp_synapses_post_nrn_id[syn_id] = metadata.post_id;
			tmp_synapses_delay[syn_id] = metadata.synapse_delay;
			tmp_synapses_delay_timer[syn_id] = -1;
			tmp_synapses_weight[syn_id] = metadata.synapse_weight;
			syn_id++;
		}

		synapses_number[neuron_id] = syn_count;

		cudaMalloc((void**)&synapses_post_nrn_id[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay_timer[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_weight[neuron_id], datasize<float>(syn_count));

		cudaMemcpy(synapses_post_nrn_id[neuron_id], &tmp_synapses_post_nrn_id, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay[neuron_id], &tmp_synapses_delay, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay_timer[neuron_id], &tmp_synapses_delay_timer, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_weight[neuron_id], &tmp_synapses_weight, datasize<float>(syn_count), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void ***)&gpu_synapses_post_nrn_id, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_post_nrn_id, synapses_post_nrn_id, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_delay, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_delay, synapses_delay, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_delay_timer, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_delay_timer, synapses_delay_timer, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_weight, datasize<float *>(neurons_number));
	memcpyHtD<float *>(gpu_synapses_weight, synapses_weight, neurons_number);

	cudaMalloc(&gpu_old_v, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_old_v, old_v, neurons_number);

	cudaMalloc(&gpu_old_u, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_old_u, old_u, neurons_number);

	cudaMalloc(&gpu_has_spike, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_spike, has_spike, neurons_number);

	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);

	cudaMalloc(&gpu_nrn_ref_timer, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_nrn_ref_timer, nrn_ref_timer, neurons_number);

	cudaMalloc(&gpu_nrn_current, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_nrn_current, nrn_current, neurons_number);

	cudaMalloc(&gpu_has_generator, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_generator, has_generator, neurons_number);

	cudaMalloc(&gpu_begin_spiking, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_begin_spiking, begin_spiking, neurons_number);

	cudaMalloc(&gpu_end_spiking, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_end_spiking, end_spiking, neurons_number);

	cudaMalloc(&gpu_has_multimeter, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_multimeter, has_multimeter, neurons_number);

	cudaMalloc(&gpu_synapses_number, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_synapses_number, synapses_number, neurons_number);

	cudaMalloc(&gpu_spiking_per_step, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_spiking_per_step, spiking_per_step, neurons_number);

	cudaMalloc(&gpu_multimeter_result, datasize<float>(sim_time_in_steps));
	memcpyHtD<float>(gpu_multimeter_result, multimeter_result, sim_time_in_steps);

	// FixMe debugging functionality
	cudaMalloc(&gpu_voltage_recording, datasize<float>(neurons_number * sim_time_in_steps));
	memcpyHtD<float>(gpu_voltage_recording, voltage_recording, neurons_number * sim_time_in_steps);
	cudaMalloc(&gpu_current_recording, datasize<float>(neurons_number * sim_time_in_steps));
	memcpyHtD<float>(gpu_current_recording, current_recording, neurons_number * sim_time_in_steps);
	cudaMalloc(&gpu_spike_recording, datasize<int>(neurons_number * sim_time_in_steps));
	memcpyHtD<int>(gpu_spike_recording, spike_recording, neurons_number * sim_time_in_steps);

	int threads_per_block = 1024;
	int num_blocks = 1; //neurons_number / threads_per_block + 1;

	printf("Size of network: %i \n", neurons_number);
	printf("Start GPU with %d threads x %d blocks (Total: %d threads) \n",
		   threads_per_block, num_blocks, threads_per_block * num_blocks);

	// measure GPU ellapsed time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	sim_kernel<<<num_blocks, threads_per_block>>>(
			gpu_old_v,
					gpu_old_u,
					gpu_nrn_current,
					gpu_nrn_ref_time,
					gpu_nrn_ref_timer,
					gpu_synapses_number,
					gpu_has_spike,
					gpu_synapses_post_nrn_id,
					gpu_synapses_delay,
					gpu_synapses_delay_timer,
					gpu_synapses_weight,
					neurons_number,
					gpu_has_generator,
					gpu_has_multimeter,
					gpu_multimeter_result,
					gpu_begin_spiking,
					gpu_end_spiking,
					gpu_spiking_per_step,
					// ToDo remove after debugging
					gpu_voltage_recording,
					gpu_current_recording,
					gpu_spike_recording
	);

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double t = milliseconds / 1e3;
	double realtime_factor = T_SIMULATION / t / 1e3;
	printf("Ellapsed time: %fs. Realtime factor: x%f (%s than realtime)\n",
		   t, realtime_factor, realtime_factor > 1? "faster":"slower");

	// copy neurons/synapses array to the HOST
	memcpyDtH<float>(multimeter_result, gpu_multimeter_result, sim_time_in_steps);
	// ToDo remove after debugging
	memcpyDtH<float>(voltage_recording, gpu_voltage_recording, neurons_number * sim_time_in_steps);
	memcpyDtH<float>(current_recording, gpu_current_recording, neurons_number * sim_time_in_steps);
	memcpyDtH<int>(spike_recording, gpu_spike_recording, neurons_number * sim_time_in_steps);

	// tell the CPU to halt further processing until the CUDA kernel has finished doing its business
	cudaDeviceSynchronize();

	save_result(test_index, voltage_recording, current_recording, spike_recording, neurons_number);

	cudaDeviceReset();
}

int main() {
	simulate(0);

	return 0;
}