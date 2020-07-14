#include <algorithm>
#include <cstdio>
#include <math.h>
#include <utility>
#include <vector>
#include <ctime>
#include <stdexcept>
#include <random>
#include <curand_kernel.h>
#include <chrono>
// for file writing
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace std;

unsigned int global_id = 0;
unsigned int SIM_TIME_IN_STEPS;
const int LEG_STEPS = 1;             // [step] number of full cycle steps
const float SIM_STEP = 0.025;        // [ms] simulation step
// stuff variables
const int neurons_in_group = 50;     // number of neurons in a group
const int neurons_in_ip = 196;       // number of neurons in a group

class Group {
public:
	Group() = default;
	string group_name;
	unsigned int id_start{};
	unsigned int id_end{};
	unsigned int group_size{};
};

// struct for human-readable initialization of connectomes
struct SynapseMetadata {
	unsigned int pre_id;         // pre neuron ID
	unsigned int post_id;        // post neuron ID
	unsigned int synapse_delay;  // [step] synaptic delay of the synapse (axonal delay is included to this delay)
	float synapse_weight;        // [nS] synaptic weight. Interpreted as changing conductivity of neuron membrane

	SynapseMetadata(int pre_id, int post_id, float synapse_delay, float synapse_weight){
		this->pre_id = pre_id;
		this->post_id = post_id;
		this->synapse_delay = lround(synapse_delay * (1 / SIM_STEP) + 0.5);
		this->synapse_weight = synapse_weight;
	}
};

// struct for human-readable initialization of connectomes
struct GroupMetadata {
	Group group;
	float* g_exc;                // [nS] array of excitatory conductivity
	float* g_inh;                // [nS] array of inhibition conductivity
	float* voltage_array;        // [mV] array of membrane potential
	vector<float> spike_vector;  // [ms] spike times

	explicit GroupMetadata(Group group){
		this->group = move(group);
		voltage_array = new float[SIM_TIME_IN_STEPS];
		g_exc = new float[SIM_TIME_IN_STEPS];
		g_inh = new float[SIM_TIME_IN_STEPS];
	}
};

__host__
unsigned int ms_to_step(float ms) { return (unsigned int)(ms / SIM_STEP); }

__host__
float step_to_ms(int step) { return step * SIM_STEP; }

vector<GroupMetadata> all_groups;
vector<SynapseMetadata> all_synapses;

// form structs of neurons global ID and groups name
Group form_group(const string& group_name, int nrns_in_group = neurons_in_group) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = global_id;        // first ID in the group
	group.id_end = global_id + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group

	all_groups.emplace_back(group);

	global_id += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n", group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

__device__
float dn(float V, float n) {
	float a = (10.0 - V) / (100 * (exp((10.0 - V) / 10.0) - 1.0));
	float b = 0.125 * exp(-V / 80.0);
	b = a - (a + b) * n;
	if (b != b) return 0;
	if (b + n >= 1) return 0.99 - n;
	return b;
}

__device__
float dh(float V, float h) {
	float a = 0.07 * exp(-V / 20.0);
	float b = 1.0 / (exp((30.0 - V) / 10.0) + 1.0);
	b = a - (a + b) * h;
	if (b != b) return 0;
	if (b + h >= 1) return 0.99 - h;
	return b;
}

__device__
float dm(float V, float m) {
	float a = (25.0 - V) / (10.0 * (exp((25.0 - V) / 10.0) - 1.0));
	float b = 4.0 * exp(-V / 18.0);
	b = a - (a + b) * m;
	if (b != b) return 0;
	if (b + m >= 1) return 0.99 - m;
	return b;
}

__global__
void neurons_kernel(float *V_extra,
                    float *V_in,
                    float *V_mid,
                    float *V_out,
                    float *h_in,
                    float *h_mid,
                    float *h_out,
                    float *m_in,
                    float *m_mid,
                    float *m_out,
                    float *n_in,
                    float *n_mid,
                    float *n_out,
                    const float *g_Na,
                    const float *g_K,
                    const float *g_L,
                    float *g_exc,
                    float *g_inh,
                    const float *const_coef1,
                    const float *const_coef2,
                    const float *const_coef3,
                    bool *has_spike,
                    const unsigned short *nrn_ref_time,
                    unsigned short *nrn_ref_time_timer,
                    const int neurons_number,
                    const short EES_activated,
                    const short CV_activated,
                    const bool C0_activated,
                    const bool C0_early_activated,
                    const unsigned int sim_iter,
                    const int decrease_lvl_Ia_spikes,
                    const float sim_step){
	/// neuron parameters
	const float E_Na = 50.0;         // [mV] Reversal potential for the Sodium current
	const float E_K = -90.0;         // [mV] Reversal potential for the Potassium current
	const float E_L = -72.0;         // [mV] Reversal potential for the leak current
	const float E_ex = 50.0;         // [mV] Reversal potential for excitatory input
	const float E_in = -80.0;        // [mV] Reversal potential for inhibitory input
	const float tau_syn_exc = 0.2;   // [ms] Decay time of excitatory synaptic current (ms)
	const float tau_syn_inh = 2.0;   // [ms] Decay time of inhibitory synaptic current (ms)
	const float V_adj = -65.0;       // adjusts threshold to around -50 mV
	const float g_bar = 5;           // [nS] the maximal possible conductivity

	float I_syn_exc, I_syn_inh;
	float I_K, I_Na, I_L, V_out_old, dV_mid;

	/// STRIDE neuron update
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < neurons_number; tid += blockDim.x * gridDim.x) {
		// reset spike flag of the current neuron before calculations
		has_spike[tid] = false;
		// Ia_E/F_aff IDs [1947 ... 2066] [2067 ... 2186] control spike number of Ia afferent by resetting neuron current
		if (1947 <= tid && tid <= 2186) {
			// reset current of 1/3 of neurons
			if (decrease_lvl_Ia_spikes == 1 && tid % 3 == 0) {
				g_exc[tid] = 0;
			} else {
				// reset current of 1/2 of neurons
				if (decrease_lvl_Ia_spikes == 2 && tid % 2 == 0) {
					g_exc[tid] = 0;
				}
			}
		}
		// generate spikes for EES
		if (tid < 50 && EES_activated) has_spike[tid] = true;
		// skin stimulations
		if (!C0_activated) {
			if (tid == 300 && CV_activated == 1 && (sim_iter % 4 == 0)) has_spike[tid] = true;
			if (tid == 301 && CV_activated == 2 && (sim_iter % 4 == 0)) has_spike[tid] = true;
			if (tid == 302 && CV_activated == 3 && (sim_iter % 4 == 0)) has_spike[tid] = true;
			if (tid == 303 && CV_activated == 4 && (sim_iter % 4 == 0)) has_spike[tid] = true;
			if (tid == 304 && CV_activated == 5 && (sim_iter % 4 == 0)) has_spike[tid] = true;
		}
		// increased barrier for muscles
		if (3463 <= tid && tid <= 3712) {
			// the maximal value of input current
			if (g_exc[tid] > 200) g_exc[tid] = 200;
			if (g_inh[tid] > 200) g_inh[tid] = 200;
		} else {
			// the maximal value of input current
			if (g_exc[tid] > g_bar) g_exc[tid] = g_bar;
			if (g_inh[tid] > g_bar) g_inh[tid] = g_bar;
		}
		// synaptic currents
		I_syn_exc = g_exc[tid] * (V_in[tid] - E_ex);
		I_syn_inh = g_inh[tid] * (V_in[tid] - E_in);
		V_out_old = V_out[tid];

		if (3463 <= tid && tid <= 3712) {
			I_syn_exc = g_exc[tid] * (V_in[tid] - 0);
			I_syn_inh = g_inh[tid] * (V_in[tid] - E_in);
		}

		// if neuron in the refractory state -- ignore synaptic inputs. Re-calculate membrane potential
		if (nrn_ref_time_timer[tid] != 0) {//} || nrn_ref_time_timer[tid] + 10 > nrn_ref_time[tid]) {
			I_syn_exc = 0;
			I_syn_inh = 0;
		}
		// ionic currents
		I_K = g_K[tid] * pow(n_in[tid], 4) * (V_in[tid] - E_K);
		I_Na = g_Na[tid] * pow(m_in[tid], 3) * h_in[tid] * (V_in[tid] - E_Na);
		I_L = g_L[tid] * (V_in[tid] - E_L);
		V_in[tid] += const_coef2[tid] * (const_coef1[tid] * (2 * V_mid[tid] - 2 * V_in[tid]) - I_Na - I_K - I_L - I_syn_exc - I_syn_inh);
		if (V_in[tid] != V_in[tid]) V_in[tid] = -72;

		I_K = g_K[tid] * pow(n_mid[tid], 4) * (V_mid[tid] - E_K);
		I_Na = g_Na[tid] * pow(m_mid[tid], 3) * h_mid[tid] * (V_mid[tid] - E_Na);
		I_L = g_L[tid] * (V_mid[tid] - E_L);
		dV_mid = const_coef2[tid] * (const_coef1[tid] * (V_out[tid] - 2 * V_mid[tid] + V_in[tid]) - I_Na - I_K - I_L);
		V_mid[tid] += dV_mid;
		if (V_mid[tid] != V_mid[tid]) V_mid[tid] = -72;

		V_extra[tid] = const_coef3[tid] * (I_L + I_Na + I_K + (1 / const_coef2[tid]) * dV_mid);

		I_K = g_K[tid] * pow(n_out[tid], 4) * (V_out[tid] - E_K);
		I_Na = g_Na[tid] * pow(m_out[tid], 3) * h_out[tid] * (V_out[tid] - E_Na);
		I_L = g_L[tid] * (V_out[tid] - E_L);
		V_out[tid] += const_coef2[tid] * (const_coef1[tid] * (2 * V_mid[tid] - 2 * V_out[tid]) - I_Na - I_K - I_L);
		if (V_out[tid] != V_out[tid]) V_out[tid] = -72;

		// use temporary dV variable as V_m with adjust
		/// transition rates between open and closed states of the potassium channels
		n_in[tid] += dn(V_in[tid] - V_adj, n_in[tid]) * sim_step;
		n_mid[tid] += dn(V_mid[tid] - V_adj, n_mid[tid]) * sim_step;
		n_out[tid] += dn(V_out[tid] - V_adj, n_out[tid]) * sim_step;

		m_in[tid] += dm(V_in[tid] - V_adj, m_in[tid]) * sim_step;
		m_mid[tid] += dm(V_mid[tid] - V_adj, m_mid[tid]) * sim_step;
		m_out[tid] += dm(V_out[tid] - V_adj, m_out[tid]) * sim_step;

		h_in[tid] += dh(V_in[tid] - V_adj, h_in[tid]) * sim_step;
		h_mid[tid] += dh(V_mid[tid] - V_adj, h_mid[tid]) * sim_step;
		h_out[tid] += dh(V_out[tid] - V_adj, h_out[tid]) * sim_step;

		// re-calculate conductance
		g_exc[tid] -= g_exc[tid] / tau_syn_exc * sim_step;
		g_inh[tid] -= g_inh[tid] / tau_syn_inh * sim_step;

		// threshold && not in refractory period
		if (nrn_ref_time_timer[tid] == 0 && V_out[tid] >=  V_adj + 30.0 && V_out_old > V_out[tid]) {
			has_spike[tid] = true;  // set spike state. It will be used in the "synapses_kernel"
			nrn_ref_time_timer[tid] = nrn_ref_time[tid];  // set the refractory period
		}
		// update the refractory period timer
		if (nrn_ref_time_timer[tid] > 0)
			nrn_ref_time_timer[tid]--;
	}
}

__global__
void synapses_kernel(const bool *neuron_has_spike,     // array of bools -- is neuron has spike or not
                     float *neuron_g_exc,              // array of excitatory conductivity per neuron (changable)
                     float *neuron_g_inh,              // array of inhibitory conductivity per neuron (changable)
                     const int *synapses_pre_nrn_id,   // array of pre neurons ID per synapse
                     const int *synapses_post_nrn_id,  // array of post neurons ID per synapse
                     const int *synapses_delay,        // array of synaptic delay per synapse
                     int *synapses_delay_timer,        // array as above but changable
                     const float *synapses_weight,     // array of synaptic weight per synapse
                     const int syn_number){            // number of synapses
	// ignore threads which ID is greater than neurons number
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < syn_number; tid += blockDim.x * gridDim.x) {
		// add synaptic delay if neuron has spike
		if (synapses_delay_timer[tid] == -1 && neuron_has_spike[synapses_pre_nrn_id[tid]]) {
			synapses_delay_timer[tid] = synapses_delay[tid];
		}
		// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
		if (synapses_delay_timer[tid] == 0) {
			// post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
			if (synapses_weight[tid] >= 0) {
				atomicAdd(&neuron_g_exc[synapses_post_nrn_id[tid]], synapses_weight[tid]);
			} else {
				// remove negative sign
				atomicAdd(&neuron_g_inh[synapses_post_nrn_id[tid]], -synapses_weight[tid]);
			}
			// make synapse timer a "free" for next spikes
			synapses_delay_timer[tid] = -1;
		}
		// update synapse delay timer
		if (synapses_delay_timer[tid] > 0) {
			synapses_delay_timer[tid]--;
		}
	}
}


// copy data from host to device
template <typename type>
void memcpyHtD(type* gpu, type* host, unsigned int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

// copy data from device to host
template <typename type>
void memcpyDtH(type* host, type* gpu, unsigned int size) {
	cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

// get datasize of current variable type and its number
template <typename type>
unsigned int datasize(unsigned int size) {
	return sizeof(type) * size;
}

// fill array with current value
template <typename type>
void fill_array(type *array, unsigned int size, type value) {
	for(int i = 0; i < size; i++)
		array[i] = value;
}

template <typename type>
type* init_gpu_arr(type* cpu_var, int count) {
	type* gpu_var;
	cudaMalloc(&gpu_var, datasize<type>(count));
	memcpyHtD<type>(gpu_var, cpu_var, count);
	return gpu_var;
}

template <typename type>
type* init_cpu_arr(int size, type val) {
	type *array = new type[size];
	for(int i = 0; i < size; i++)
		array[i] = val;
	return array;
}

template <typename type>
type* init_cpu_arr_normal(int size, type mean, type stddev) {
	random_device r;
	default_random_engine generator(r());
	normal_distribution<float> distr(mean, stddev);

	type *array = new type[size];
	for(int i = 0; i < size; i++)
		array[i] = (type)distr(generator);
	return array;
}

int get_skin_stim_time(int cms) {
	if (cms == 21)
		return 25;
	if (cms == 15)
		return 50;
	return 125;
}

void bimodal_distr_for_moto_neurons(float *nrn_diameter) {
	int diameter_active = 27;
	int scale_active = 3;
	int diameter_standby = 57;
	int scale_standby = 6;
	// MN_E [1557 ... 1766] MN_F [1767 ... 1946]
	int MN_E_start = 1557;
	int MN_E_end = 1766;
	int MN_F_start = 1767;
	int MN_F_end = 1946;

	int nrn_number_extensor = MN_E_end - MN_E_start;
	int nrn_number_flexor = MN_F_end - MN_E_start;

	int standby_percent = 70;

	int standby_size_extensor = (int)(nrn_number_extensor * standby_percent / 100);
	int standby_size_flexor = (int)(nrn_number_flexor * standby_percent / 100);
	int active_size_extensor = nrn_number_extensor - standby_size_extensor;
	int active_size_flexor = nrn_number_flexor - standby_size_flexor;

	random_device r1;
	default_random_engine generator1(r1());
	normal_distribution<float> d_active(diameter_active, scale_active);
	normal_distribution<float> d_standby(diameter_standby, scale_standby);

	for (int i = MN_E_start; i < MN_E_start + active_size_extensor; i++) {
		nrn_diameter[i] = d_active(generator1);
	}
	for (int i = MN_E_start + active_size_extensor; i <= MN_E_end; i++) {
		nrn_diameter[i] = d_standby(generator1);
	}

	for (int i = MN_F_start; i < MN_F_start + active_size_flexor; i++) {
		nrn_diameter[i] = d_active(generator1);
	}
	for (int i = MN_F_start + active_size_flexor; i <= MN_F_end; i++) {
		nrn_diameter[i] = d_standby(generator1);
	}
}

void save(int test_index, GroupMetadata &metadata, const string& folder){
	ofstream file;
	string file_name = "/dat/" + to_string(test_index) + "_" + metadata.group.group_name + ".dat";

	file.open(folder + file_name);
	// save voltage
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.voltage_array[sim_iter] << " ";
	file << endl;

	// save g_exc
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.g_exc[sim_iter] << " ";
	file << endl;

	// save g_inh
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.g_inh[sim_iter] << " ";
	file << endl;

	// save spikes
	for (float const& value: metadata.spike_vector) {
		file << value << " ";
	}
	file.close();

	cout << "Saved to: " << folder + file_name << endl;
}

void save_result(int test_index, int save_all) {
	string current_path = getcwd(nullptr, 0);

	printf("[Test #%d] Save %s results to: %s \n", test_index, (save_all == 0)? "MOTO" : "ALL", current_path.c_str());

	for(GroupMetadata &metadata : all_groups) {
		if (save_all == 0) {
			if(metadata.group.group_name == "MN_E")
				save(test_index, metadata, current_path);
			if(metadata.group.group_name == "MN_F")
				save(test_index, metadata, current_path);
		} else {
			save(test_index, metadata, current_path);
		}
	}
}

void copy_data_to(GroupMetadata &metadata,
                  const float* nrn_v_m,
                  const float* nrn_g_exc,
                  const float* nrn_g_inh,
                  const bool *nrn_has_spike,
                  const unsigned int sim_iter) {
	float nrn_mean_volt = 0;
	float nrn_mean_g_exc = 0;
	float nrn_mean_g_inh = 0;

	for(unsigned int tid = metadata.group.id_start; tid <= metadata.group.id_end; tid++) {
		nrn_mean_volt += nrn_v_m[tid];
		nrn_mean_g_exc += nrn_g_exc[tid];
		nrn_mean_g_inh += nrn_g_inh[tid];
		if (nrn_has_spike[tid]) {
			metadata.spike_vector.push_back(step_to_ms(sim_iter) + 0.25);
		}
	}
	metadata.g_exc[sim_iter] = nrn_mean_g_exc / metadata.group.group_size;
	metadata.g_inh[sim_iter] = nrn_mean_g_inh / metadata.group.group_size;
	metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size;
}

void connect_one_to_all(const Group& pre_neurons, const Group& post_neurons, float syn_delay, float weight) {
	// Seed with a real random value, if available
	random_device r;
	default_random_engine generator(r());
	normal_distribution<float> delay_distr(syn_delay, syn_delay / 5);
	normal_distribution<float> weight_distr(weight, weight / 10);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (unsigned int post_id = post_neurons.id_start; post_id <= post_neurons.id_end; post_id++) {
			all_synapses.emplace_back(pre_id, post_id, delay_distr(generator), weight_distr(generator));
		}
	}

	printf("Connect %s to %s [one_to_all] (1:%d). Total: %d W=%.2f, D=%.1f\n", pre_neurons.group_name.c_str(),
	       post_neurons.group_name.c_str(), post_neurons.group_size, pre_neurons.group_size * post_neurons.group_size,
	       weight, syn_delay);
}

void connect_fixed_outdegree(const Group& pre_neurons,
                             const Group& post_neurons,
                             float syn_delay,
                             float syn_weight,
                             int outdegree=0,
                             bool no_distr=false) {
	// connect neurons with uniform distribution and normal distributon for syn delay and syn_weight
	random_device r;
	default_random_engine generator(r());
	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	uniform_int_distribution<int> outdegree_num(30, 50);
	normal_distribution<float> delay_distr_gen(syn_delay, syn_delay / 5);
	normal_distribution<float> weight_distr_gen(syn_weight, syn_weight / 10);

	if (outdegree == 0)
		outdegree = outdegree_num(generator);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = id_distr(generator);
			float syn_delay_distr = delay_distr_gen(generator);
			if (syn_delay_distr <= 0.2) {
				syn_delay_distr = 0.2;
			}
			float syn_weight_distr = weight_distr_gen(generator);
			if (no_distr) {
				all_synapses.emplace_back(pre_id, rand_post_id, syn_delay, syn_weight);
			} else {
				all_synapses.emplace_back(pre_id, rand_post_id, syn_delay_distr, syn_weight_distr);
			}
		}
	}

	printf("Connect %s to %s [fixed_outdegree] (1:%d). Total: %d W=%.2f, D=%.1f\n",
	       pre_neurons.group_name.c_str(), post_neurons.group_name.c_str(),
	       outdegree, pre_neurons.group_size * outdegree, syn_weight, syn_delay);
}

void init_network() {
	/// groups of neurons
	Group EES = form_group("EES");
	Group E1 = form_group("E1");
	Group E2 = form_group("E2");
	Group E3 = form_group("E3");
	Group E4 = form_group("E4");
	Group E5 = form_group("E5");

	Group CV1 = form_group("CV1", 1);
	Group CV2 = form_group("CV2", 1);
	Group CV3 = form_group("CV3", 1);
	Group CV4 = form_group("CV4", 1);
	Group CV5 = form_group("CV5", 1);
	Group CD4 = form_group("CD4", 1);
	Group CD5 = form_group("CD5", 1);

	Group OM1_0 = form_group("OM1_0");
	Group OM1_1 = form_group("OM1_1");
	Group OM1_2_E = form_group("OM1_2_E");
	Group OM1_2_F = form_group("OM1_2_F");
	Group OM1_3 = form_group("OM1_3");

	Group OM2_0 = form_group("OM2_0");
	Group OM2_1 = form_group("OM2_1");
	Group OM2_2_E = form_group("OM2_2_E");
	Group OM2_2_F = form_group("OM2_2_F");
	Group OM2_3 = form_group("OM2_3");

	Group OM3_0 = form_group("OM3_0");
	Group OM3_1 = form_group("OM3_1");
	Group OM3_2_E = form_group("OM3_2_E");
	Group OM3_2_F = form_group("OM3_2_F");
	Group OM3_3 = form_group("OM3_3");

	Group OM4_0 = form_group("OM4_0");
	Group OM4_1 = form_group("OM4_1");
	Group OM4_2_E = form_group("OM4_2_E");
	Group OM4_2_F = form_group("OM4_2_F");
	Group OM4_3 = form_group("OM4_3");

	Group OM5_0 = form_group("OM5_0");
	Group OM5_1 = form_group("OM5_1");
	Group OM5_2_E = form_group("OM5_2_E");
	Group OM5_2_F = form_group("OM5_2_F");
	Group OM5_3 = form_group("OM5_3");

	Group MN_E = form_group("MN_E", 210);
	Group MN_F = form_group("MN_F", 180);

	Group Ia_E_aff = form_group("Ia_E_aff", 120);
	Group Ia_F_aff = form_group("Ia_F_aff", 120);

	Group R_E = form_group("R_E");
	Group R_F = form_group("R_F");

	Group Ia_E_pool = form_group("Ia_E_pool", neurons_in_ip);
	Group Ia_F_pool = form_group("Ia_F_pool", neurons_in_ip);

	Group eIP_E = form_group("eIP_E", neurons_in_ip);
	Group eIP_F = form_group("eIP_F", neurons_in_ip);

	Group iIP_E = form_group("iIP_E", neurons_in_ip);
	Group iIP_F = form_group("iIP_F", neurons_in_ip);

	Group muscle_E = form_group("muscle_E", 150);
	Group muscle_F = form_group("muscle_F", 100);

	/// connectomes
	connect_fixed_outdegree(EES, E1, 1, 5);
	connect_fixed_outdegree(E1, E2, 1, 2);
	connect_fixed_outdegree(E2, E3, 1, 2);
	connect_fixed_outdegree(E3, E4, 1, 2);
	connect_fixed_outdegree(E4, E5, 1, 2);

	connect_one_to_all(CV3, OM1_3, 0.1, 0.3);
	connect_one_to_all(CV4, OM1_3, 0.1, 0.3);
	connect_one_to_all(CV5, OM1_3, 0.1, 0.3);
	connect_one_to_all(CV4, OM2_3, 0.1, 0.3);
	connect_one_to_all(CV5, OM2_3, 0.1, 0.3);
	connect_one_to_all(CV5, OM3_3, 0.1, 0.3);

	connect_one_to_all(OM1_2_E, eIP_E, 1.5, 1);
	connect_one_to_all(OM2_2_E, eIP_E, 2, 1);
	connect_one_to_all(OM3_2_E, eIP_E, 3, 1);
	connect_one_to_all(OM4_2_E, eIP_E, 4, 2);
	connect_one_to_all(OM5_2_E, eIP_E, 4.5, 2);

	connect_fixed_outdegree(E1, OM1_0, 1, 0.5);
	// input from sensory
	connect_one_to_all(CV1, OM1_0, 0.5, 0.3);
	connect_one_to_all(CV2, OM1_0, 0.5, 0.3);
	// inner connectomes
	connect_fixed_outdegree(OM1_0, OM1_1, 1, 0.3);
	connect_fixed_outdegree(OM1_1, OM1_2_E, 2, 0.6);
	connect_fixed_outdegree(OM1_1, OM1_3, 2, 0.07);
	connect_fixed_outdegree(OM1_2_E, OM1_1, 2.5, 0.8);
	connect_fixed_outdegree(OM1_2_E, OM1_2_E, 2.5, 0.8);
	connect_fixed_outdegree(OM1_2_E, OM1_3, 1, 0.07);
	connect_fixed_outdegree(OM1_3, OM1_1, 3, -0.1);
	connect_fixed_outdegree(OM1_3, OM1_2_E, 3, -0.1);

	connect_fixed_outdegree(E2, OM2_0, 1, 0.1);
	// input from sensory
	connect_one_to_all(CV2, OM2_0, 0.5, 0.3);
	connect_one_to_all(CV3, OM2_0, 0.5, 0.3);
	// inner connectomes
	connect_fixed_outdegree(OM2_0, OM2_1, 1, 0.3);
	connect_fixed_outdegree(OM2_1, OM2_2_E, 2, 0.6);
	connect_fixed_outdegree(OM2_1, OM2_3, 2, 0.1);
	connect_fixed_outdegree(OM2_2_E, OM2_1, 2.5, 0.6);
	connect_fixed_outdegree(OM2_2_E, OM2_2_E, 2.5, 0.3);
	connect_fixed_outdegree(OM2_2_E, OM2_3, 1, 0.1);
	connect_fixed_outdegree(OM2_3, OM2_1, 1, -0.1);
	connect_fixed_outdegree(OM2_3, OM2_2_E, 1, -0.1);
	///
	connect_fixed_outdegree(E3, OM3_0, 1, 0.1);
	// input from sensory
	connect_one_to_all(CV3, OM3_0, 0.5, 0.3);
	connect_one_to_all(CV4, OM3_0, 0.5, 0.3);
	// inner connectomes
	connect_fixed_outdegree(OM3_0, OM3_1, 1, 0.3);
	connect_fixed_outdegree(OM3_1, OM3_2_E, 2, 0.7);
	connect_fixed_outdegree(OM3_1, OM3_3, 2, 0.1);
	connect_fixed_outdegree(OM3_2_E, OM3_1, 2.5, 0.6);
	connect_fixed_outdegree(OM3_2_E, OM3_2_E, 2.5, 0.3);
	connect_fixed_outdegree(OM3_2_E, OM3_3, 1, 0.1);
	connect_fixed_outdegree(OM3_3, OM3_1, 1, -0.1);
	connect_fixed_outdegree(OM3_3, OM3_2_E, 1, -0.1);
	///
	connect_fixed_outdegree(E4, OM4_0, 1, 0.1);
	// input from sensory
	connect_one_to_all(CV4, OM4_0, 0.5, 0.3);
	connect_one_to_all(CV5, OM4_0, 0.5, 0.3);
	// inner connectomes
	connect_fixed_outdegree(OM4_0, OM4_1, 1, 0.3);
	connect_fixed_outdegree(OM4_1, OM4_2_E, 2, 0.7);
	connect_fixed_outdegree(OM4_1, OM4_3, 2, 0.1);
	connect_fixed_outdegree(OM4_2_E, OM4_1, 2.5, 0.6);
	connect_fixed_outdegree(OM4_2_E, OM4_2_E, 2.5, 0.4);
	connect_fixed_outdegree(OM4_2_E, OM4_3, 1, 0.1);
	connect_fixed_outdegree(OM4_3, OM4_1, 1, -0.1);
	connect_fixed_outdegree(OM4_3, OM4_2_E, 1, -0.1);
	///
	connect_fixed_outdegree(E5, OM5_0, 1, 0.1);
	// input from sensory
	connect_one_to_all(CV5, OM5_0, 0.5, 0.3);
	// inner connectomes
	connect_fixed_outdegree(OM5_0, OM5_1, 1, 0.3);
	connect_fixed_outdegree(OM5_1, OM5_2_E, 2, 0.6);
	connect_fixed_outdegree(OM5_1, OM5_3, 2, 0.1);
	connect_fixed_outdegree(OM5_2_E, OM5_1, 2.5, 0.6);
	connect_fixed_outdegree(OM5_2_E, OM5_2_E, 2.5, 0.4);
	connect_fixed_outdegree(OM5_2_E, OM5_3, 1, 0.2);
	connect_fixed_outdegree(OM5_3, OM5_1, 1, -0.1);
	connect_fixed_outdegree(OM5_3, OM5_2_E, 1, -0.1);

	/// reflex arc
	connect_fixed_outdegree(iIP_E, eIP_F, 0.5, -1);
	connect_fixed_outdegree(iIP_F, eIP_E, 0.5, -1);

	connect_fixed_outdegree(iIP_E, OM1_2_F, 0.5, -0.5);
	connect_fixed_outdegree(iIP_E, OM2_2_F, 0.5, -0.5);
	connect_fixed_outdegree(iIP_E, OM3_2_F, 0.5, -0.5);
	connect_fixed_outdegree(iIP_E, OM4_2_F, 0.5, -0.5);

	connect_fixed_outdegree(EES, Ia_E_aff, 1, 10);
	connect_fixed_outdegree(EES, Ia_F_aff, 1, 10);

	connect_fixed_outdegree(eIP_E, MN_E, 3, 13); // 2.2
	connect_fixed_outdegree(eIP_F, MN_F, 3, 13);

	connect_fixed_outdegree(iIP_E, Ia_E_pool, 1, 1);
	connect_fixed_outdegree(iIP_F, Ia_F_pool, 1, 1);

	connect_fixed_outdegree(Ia_E_pool, MN_F, 1, -1);
	connect_fixed_outdegree(Ia_E_pool, Ia_F_pool, 1, -1);
	connect_fixed_outdegree(Ia_F_pool, MN_E, 1, -1);
	connect_fixed_outdegree(Ia_F_pool, Ia_E_pool, 1, -1);

	connect_fixed_outdegree(Ia_E_aff, MN_E, 2, 50);
	connect_fixed_outdegree(Ia_F_aff, MN_F, 2, 50);

	connect_fixed_outdegree(MN_E, R_E, 2, 1);
	connect_fixed_outdegree(MN_F, R_F, 2, 1);

	connect_fixed_outdegree(MN_E, muscle_E, 2, 100);
	connect_fixed_outdegree(MN_F, muscle_F, 2, 100);
	connect_fixed_outdegree(EES, muscle_E, 1, 150);

	connect_fixed_outdegree(R_E, MN_E, 2, -0.5);
	connect_fixed_outdegree(R_E, R_F, 2, -1);

	connect_fixed_outdegree(R_F, MN_F, 2, -0.5);
	connect_fixed_outdegree(R_F, R_E, 2, -1);
}

__host__
void simulate(int cms, int ees, int inh, int ped, int ht5, int save_all, int itest) {
	// init random distributions
	random_device r;
	default_random_engine generator(r());
	uniform_real_distribution<float> standard_uniform(0, 1);
	uniform_real_distribution<float> d_inter_distr(3, 8);
	uniform_real_distribution<float> d_Ia_aff_distr(10, 20);
	uniform_real_distribution<float> d_muscle_dist(40, 50);

	const unsigned int skin_stim_time = get_skin_stim_time(cms);
	const unsigned int T_simulation = 11 * skin_stim_time * LEG_STEPS;
	printf("T: %d\n", T_simulation);
	// calculate how much steps in simulation time [steps]
	SIM_TIME_IN_STEPS = ms_to_step(T_simulation);
	// calculate spike frequency and C0/C1 activation time in steps
	auto ees_spike_each_step = ms_to_step(1000 / ees);
	auto steps_activation_C0 = ms_to_step(5 * skin_stim_time);
	auto steps_activation_C1 = ms_to_step(6 * skin_stim_time);

	/// init neuron groups and connectomes
	init_network();

	// get the number of bio objects
	const auto neurons_number = global_id;
	const auto synapses_number = static_cast<int>(all_synapses.size());
	/// CPU variables
	auto *nrn_v_extra = init_cpu_arr<float>(neurons_number, 0);      // [mV] neuron extracellular membrane potential
	auto *nrn_v_m_in = init_cpu_arr<float>(neurons_number, -72.5);   // [mV] input neuron intracellular membrane potential
	auto *nrn_v_m_mid = init_cpu_arr<float>(neurons_number, -72.5);  // [mV] medial neuron intracellular membrane potential
	auto *nrn_v_m_out = init_cpu_arr<float>(neurons_number, -72.5);  // [mV] output neuron intracellular membrane potential
	auto *nrn_n_in = init_cpu_arr<float>(neurons_number, 0.01);      // [0..1] potassium channel activation probability
	auto *nrn_n_mid = init_cpu_arr<float>(neurons_number, 0.01);     // --//--
	auto *nrn_n_out = init_cpu_arr<float>(neurons_number, 0.01);     // --//--
	auto *nrn_h_in = init_cpu_arr<float>(neurons_number, 0.99);      // [0..1] sodium channel activation probability
	auto *nrn_h_mid = init_cpu_arr<float>(neurons_number, 0.99);     // --//--
	auto *nrn_h_out = init_cpu_arr<float>(neurons_number, 0.99);     // --//--
	auto *nrn_m_in = init_cpu_arr<float>(neurons_number, 0.01);      // [0..1] sodium channel inactivation probability
	auto *nrn_m_mid = init_cpu_arr<float>(neurons_number, 0.01);     // --//--
	auto *nrn_m_out = init_cpu_arr<float>(neurons_number, 0.01);     // --//--
	auto *const_coef1 = init_cpu_arr<float>(neurons_number, 0);      // 2.25 * d / Ra
	auto *const_coef2 = init_cpu_arr<float>(neurons_number, 0);      // dt / Cm
	auto *const_coef3 = init_cpu_arr<float>(neurons_number, 0);      //
	auto *nrn_g_Na = init_cpu_arr<float>(neurons_number, 0);         // [nS]
	auto *nrn_g_K = init_cpu_arr<float>(neurons_number, 0);          // [nS]
	auto *nrn_g_L = init_cpu_arr<float>(neurons_number, 0);          // [nS]
	auto *nrn_g_exc = init_cpu_arr<float>(neurons_number, 0);        // [nS] excitatory synapse exponential conductance
	auto *nrn_g_inh = init_cpu_arr<float>(neurons_number, 0);        // [nS] inhibitory synapse exponential conductance
	auto *nrn_diameter = init_cpu_arr<float>(neurons_number, 0);     // [um] neuron diameter
	auto *nrn_has_spike = init_cpu_arr<bool>(neurons_number, false);  // neuron state - has spike or not
	auto *nrn_ref_time_timer = init_cpu_arr<unsigned short>(neurons_number, 0);  // [step] neuron refractory time timer
	auto *nrn_ref_time = init_cpu_arr_normal<unsigned short>(neurons_number, 3 / SIM_STEP, 0.4 / SIM_STEP);   // [step] neuron refractory time

	const unsigned int beg_C_spiking[5] = {ms_to_step(0),
	                                       ms_to_step(skin_stim_time),
	                                       ms_to_step(2 * skin_stim_time),
	                                       ms_to_step(3 * skin_stim_time),
	                                       ms_to_step(5 * skin_stim_time)};
	const unsigned int end_C_spiking[5] = {ms_to_step(skin_stim_time - 0.1),
	                                       ms_to_step(2 * skin_stim_time - 0.1),
	                                       ms_to_step(3 * skin_stim_time - 0.1),
	                                       ms_to_step(5 * skin_stim_time - 0.1),
	                                       ms_to_step(6 * skin_stim_time - 0.1)};

	/// Fill the arrays
	// set by default inter neuron's diameter for all neurons
	for(int i = 0; i < neurons_number; i++)
		nrn_diameter[i] = d_inter_distr(generator);

	// set for EES, E1, E2, E3, E4, E5 constant diameter
	for (int i = 0; i < 300; i++)
		nrn_diameter[i] = 5;

	// fill array of Ia_aff neuron's diameters
	for (int i = 1947; i < 2186; i++)
		nrn_diameter[i] = d_Ia_aff_distr(generator);

	// set bimodal distribution for motoneurons
	bimodal_distr_for_moto_neurons(nrn_diameter);

	for (int i = 3463; i <= 3712; i++)
		nrn_diameter[i] = d_muscle_dist(generator);
	// S * cm-2 = 10 nS um-2
	// set C_m, g_Na, g_K, g_L arrays based on the neuron's diameters
	float S;
	for(int i = 0; i < neurons_number; i++) {
		// S = 2 Pi r h
		S = M_PI * nrn_diameter[i] * nrn_diameter[i] / 3;
		// const_coef1 = 2.25 * d / Ra
		const_coef1[i] = 2.25 * nrn_diameter[i] / 100;
		// const_coef2 = dt / Cm
		if (1557 <= i && i <= 1946) {
			const_coef2[i] = SIM_STEP / (S * 0.02);
		} else {
			// muscles
			if (3463 <= i && i <= 3712) {
				// Pi * d * l, l = 3mm = 3000um
				S = M_PI * nrn_diameter[i] * 3000;
				const_coef1[i] = 2.25 * nrn_diameter[i] / 100;
				const_coef2[i] = SIM_STEP / (S * 0.036);
			} else {
				const_coef2[i] = SIM_STEP / (S * 0.01);
			}
		}
		// sigma
		const_coef3[i] = 0.0003 * log((sqrt(10) + 1) / (sqrt(10) - 1)) / (4 * M_PI * nrn_diameter[i] * 0.3);
		nrn_g_K[i] = S * 0.4; // 0.07
		nrn_g_L[i] = S * 0.02; // 0.01
		nrn_g_Na[i] = S * 2.5; // 0.19

		// muscles
		if (3463 <= i && i <= 3712) {
			const_coef3[i] = 0.0003 * log((sqrt(10) + 1) / (sqrt(10) - 1)) / (4 * M_PI * nrn_diameter[i] * 0.3);
			nrn_g_K[i] = S * 0.1; // 0.07
			nrn_g_L[i] = S * 0.1; // 0.01
			nrn_g_Na[i] = S * 5.5; // 0.19
		}
	}

	// synapse variables
	auto *synapses_pre_nrn_id = init_cpu_arr<int>(synapses_number, 0);
	auto *synapses_post_nrn_id = init_cpu_arr<int>(synapses_number, 0);
	auto *synapses_weight = init_cpu_arr<float>(synapses_number, 0);
	auto *synapses_delay = init_cpu_arr<int>(synapses_number, 0);
	auto *synapses_delay_timer = init_cpu_arr<int>(synapses_number, -1);

	// fill arrays of synapses
	unsigned int syn_id = 0;
	for(SynapseMetadata metadata : all_synapses) {
		synapses_pre_nrn_id[syn_id] = metadata.pre_id;
		synapses_post_nrn_id[syn_id] = metadata.post_id;
		synapses_delay[syn_id] = metadata.synapse_delay;
		synapses_weight[syn_id] = metadata.synapse_weight;
		syn_id++;
	}
	all_synapses.clear();

	// neuron variables
	auto *gpu_nrn_v_extra = init_gpu_arr<float>(nrn_v_extra, neurons_number);
	auto *gpu_nrn_v_m_in = init_gpu_arr<float>(nrn_v_m_in, neurons_number);
	auto *gpu_nrn_v_m_mid = init_gpu_arr<float>(nrn_v_m_mid, neurons_number);
	auto *gpu_nrn_v_m_out = init_gpu_arr<float>(nrn_v_m_out, neurons_number);
	auto *gpu_nrn_n_in = init_gpu_arr<float>(nrn_n_in, neurons_number);
	auto *gpu_nrn_n_mid = init_gpu_arr<float>(nrn_n_mid, neurons_number);
	auto *gpu_nrn_n_out = init_gpu_arr<float>(nrn_n_out, neurons_number);
	auto *gpu_nrn_h_in = init_gpu_arr<float>(nrn_h_in, neurons_number);
	auto *gpu_nrn_h_mid = init_gpu_arr<float>(nrn_h_mid, neurons_number);
	auto *gpu_nrn_h_out = init_gpu_arr<float>(nrn_h_out, neurons_number);
	auto *gpu_nrn_m_in = init_gpu_arr<float>(nrn_m_in, neurons_number);
	auto *gpu_nrn_m_mid = init_gpu_arr<float>(nrn_m_mid, neurons_number);
	auto *gpu_nrn_m_out = init_gpu_arr<float>(nrn_m_out, neurons_number);
	auto *gpu_nrn_g_Na = init_gpu_arr<float>(nrn_g_Na, neurons_number);
	auto *gpu_nrn_g_K = init_gpu_arr<float>(nrn_g_K, neurons_number);
	auto *gpu_nrn_g_L = init_gpu_arr<float>(nrn_g_L, neurons_number);
	auto *gpu_nrn_g_exc = init_gpu_arr<float>(nrn_g_exc, neurons_number);
	auto *gpu_nrn_g_inh = init_gpu_arr<float>(nrn_g_inh, neurons_number);
	auto *gpu_const_coef1 = init_gpu_arr<float>(const_coef1, neurons_number);
	auto *gpu_const_coef2 = init_gpu_arr<float>(const_coef2, neurons_number);
	auto *gpu_const_coef3 = init_gpu_arr<float>(const_coef3, neurons_number);
	auto *gpu_nrn_has_spike = init_gpu_arr<bool>(nrn_has_spike, neurons_number);
	auto *gpu_nrn_ref_time = init_gpu_arr<unsigned short>(nrn_ref_time, neurons_number);
	auto *gpu_nrn_ref_time_timer = init_gpu_arr<unsigned short>(nrn_ref_time_timer, neurons_number);

	// synapse variables
	auto *gpu_syn_pre_nrn_id = init_gpu_arr<int>(synapses_pre_nrn_id, synapses_number);
	auto *gpu_syn_post_nrn_id = init_gpu_arr<int>(synapses_post_nrn_id, synapses_number);
	auto *gpu_syn_weight = init_gpu_arr<float>(synapses_weight, synapses_number);
	auto *gpu_syn_delay = init_gpu_arr<int>(synapses_delay, synapses_number);
	auto *gpu_syn_delay_timer = init_gpu_arr<int>(synapses_delay_timer, synapses_number);

	/// preparations for simulation
	float time;
	int local_iter = 0;
	bool C0_activated = false;
	bool C0_early_activated = false;
	short CV_activated;
	bool EES_activated;
	int shift_time_by_step = 0;
	int decrease_lvl_Ia_spikes;
	int shifted_iter_time = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// the main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		CV_activated = 0;
		decrease_lvl_Ia_spikes = 0;
		EES_activated = (sim_iter % ees_spike_each_step == 0);

		// if flexor C0 activated, find the end of it and change to C1
		if (C0_activated) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				C0_activated = false;
				local_iter = 0;
				shift_time_by_step += steps_activation_C0;
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C0 == 0)
				C0_early_activated = false;
			// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				C0_activated = true;
				local_iter = 0;
				shift_time_by_step += steps_activation_C1;
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C1 == 0)
				C0_early_activated = true;
		}

		shifted_iter_time = sim_iter - shift_time_by_step;

		if ((beg_C_spiking[0] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[0])) CV_activated = 1;
		if ((beg_C_spiking[1] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[1])) CV_activated = 2;
		if ((beg_C_spiking[2] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[2])) CV_activated = 3;
		if ((beg_C_spiking[3] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[3])) CV_activated = 4;
		if ((beg_C_spiking[4] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[4])) CV_activated = 5;

		if (CV_activated == 1) decrease_lvl_Ia_spikes = 2;
		if (CV_activated == 2) decrease_lvl_Ia_spikes = 1;
		if (CV_activated == 3) decrease_lvl_Ia_spikes = 0;
		if (CV_activated == 4) decrease_lvl_Ia_spikes = 1;
		if (CV_activated == 5) decrease_lvl_Ia_spikes = 2;

		// update local iter (warning: can be resetted at C0/C1 activation)
		local_iter++;

		// invoke GPU kernel for neurons
		neurons_kernel<<<32, 128>>>(gpu_nrn_v_extra,
		                            gpu_nrn_v_m_in,
		                            gpu_nrn_v_m_mid,
		                            gpu_nrn_v_m_out,
		                            gpu_nrn_h_in,
		                            gpu_nrn_h_mid,
		                            gpu_nrn_h_out,
		                            gpu_nrn_m_in,
		                            gpu_nrn_m_mid,
		                            gpu_nrn_m_out,
		                            gpu_nrn_n_in,
		                            gpu_nrn_n_mid,
		                            gpu_nrn_n_out,
		                            gpu_nrn_g_Na,
		                            gpu_nrn_g_K,
		                            gpu_nrn_g_L,
		                            gpu_nrn_g_exc,
		                            gpu_nrn_g_inh,
		                            gpu_const_coef1,
		                            gpu_const_coef2,
		                            gpu_const_coef3,
		                            gpu_nrn_has_spike,
		                            gpu_nrn_ref_time,
		                            gpu_nrn_ref_time_timer,
		                            neurons_number,
		                            EES_activated,
		                            CV_activated,
		                            C0_activated,
		                            C0_early_activated,
		                            sim_iter,
		                            decrease_lvl_Ia_spikes,
		                            SIM_STEP);

		// copy data from GPU
		memcpyDtH<float>(nrn_v_m_mid, gpu_nrn_v_m_mid, neurons_number);
		memcpyDtH<float>(nrn_g_exc, gpu_nrn_g_exc, neurons_number);
		memcpyDtH<float>(nrn_g_inh, gpu_nrn_g_inh, neurons_number);
		memcpyDtH<bool>(nrn_has_spike, gpu_nrn_has_spike, neurons_number);

		// fill records arrays
		for(GroupMetadata &metadata : all_groups) {
			if (save_all == 0) {
				if (metadata.group.group_name == "MN_E")
					copy_data_to(metadata, nrn_v_m_mid, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
				if (metadata.group.group_name == "MN_F")
					copy_data_to(metadata, nrn_v_m_mid, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
			} else
				copy_data_to(metadata, nrn_v_m_mid, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
		}

		// invoke GPU kernel for synapses
		synapses_kernel<<<32, 128>>>(gpu_nrn_has_spike,
		                             gpu_nrn_g_exc,
		                             gpu_nrn_g_inh,
		                             gpu_syn_pre_nrn_id,
		                             gpu_syn_post_nrn_id,
		                             gpu_syn_delay,
		                             gpu_syn_delay_timer,
		                             gpu_syn_weight,
		                             synapses_number);
	} /// end of the simulation iteration loop

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time: %d \n", (int)time);

	cudaDeviceSynchronize();  // tell the CPU to halt further processing until the CUDA has finished doing its business
	cudaDeviceReset();        // remove all all device allocations (destroy a CUDA context)

	save_result(itest, save_all);
}

// runner
int main(int argc, char* argv[]) {
	simulate(21, 40, 100, 2, 0, 1, 0);
	return 0;
}
