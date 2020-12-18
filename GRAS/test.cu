/**
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011).
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press.
DOI:10.1017/CBO9780511975899

Based on the NEURON repository
*/

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "test.h"
#include <stdexcept>
#define CHECK( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

using namespace std;

const float dt = 0.025;     // [ms] simulation step
const int sim_time = 50;    // [ms] simulation time
const auto SIM_TIME_IN_STEPS = (unsigned int)(sim_time / dt);  // [steps] converted time into steps

const bool DEBUG = false;
const bool EXTRACELLULAR = false;
const string GENERATOR = "generator";
const string INTER = "interneuron";
const string MOTO = "motoneuron";
const string MUSCLE = "muscle";

unsigned int nrns_number = 0;        // [id] global neuron id = number of neurons
unsigned int nrns_and_segs = 0;      // [id] global neuron+segs id = number of neurons with segments
unsigned int generators_id_end = 0;  // [id] id of the last generator (to avoid them for updating)
const int LEG_STEPS = 1;             // [step] number of full cycle steps

const int neurons_in_group = 50;     // number of neurons in a group
const int neurons_in_ip = 196;       // number of neurons in a group

const int skin_time = 25;  // duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
int cv_fr = 200;     // frequency of CV
int ees_fr = 100;     // frequency of EES

float cv_int = 1000 / cv_fr;
float ees_int = 1000 / ees_fr;

/*
EES_stimulus = (np.arange(0, sim_time, ees_int) / dt).astype(int)
CV1_stimulus = (np.arange(skin_time * 0, skin_time * 1, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV2_stimulus = (np.arange(skin_time * 1, skin_time * 2, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV3_stimulus = (np.arange(skin_time * 2, skin_time * 3, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV4_stimulus = (np.arange(skin_time * 3, skin_time * 5, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV5_stimulus = (np.arange(skin_time * 5, skin_time * 6, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
*/

/*
# arrays for saving data
spikes = []             # saved spikes
GRAS_data = []          # saved gras data (DEBUGGING)
save_groups = []        # neurons groups that need to save
saved_voltage = []      # saved voltage
save_neuron_ids = []    # neurons id that need to save
 */

// common neuron constants
const float k = 0.017;           // synaptic coef
const float V_th = -40;          // [mV] voltage threshold
const float V_adj = -63;         // [mV] adjust voltage for -55 threshold
// moto neuron constants
const float ca0 = 2;             // initial calcium concentration
const float amA = 0.4;           // const ??? todo
const float amB = 66;            // const ??? todo
const float amC = 5;             // const ??? todo
const float bmA = 0.4;           // const ??? todo
const float bmB = 32;            // const ??? todo
const float bmC = 5;             // const ??? todo
const float R_const = 8.314472;  // [k-mole] or [joule/degC] const
const float F_const = 96485.34;  // [faraday] or [kilocoulombs] const
// muscle fiber constants
const float g_kno = 0.01;        // [S/cm2] conductance of the todo
const float g_kir = 0.03;        // [S/cm2] conductance of the Inwardly Rectifying Potassium K+ (Kir) channel
// Boltzman steady state curve
const float vhalfl = -98.92;     // [mV] inactivation half-potential
const float kl = 10.89;          // [mV] Stegen et al. 2012
// tau_infty
const float vhalft = 67.0828;    // [mV] fitted //100 uM sens curr 350a, Stegen et al. 2012
const float at = 0.00610779;     // [/ ms] Stegen et al. 2012
const float bt = 0.0817741;      // [/ ms] Note: typo in Stegen et al. 2012
// temperature dependence
const float q10 = 1;             // temperature scaling (sensitivity)
const float celsius = 36;        // [degC] temperature of the cell
// i_membrane [mA/cm2]
const float e_extracellular = 0; // [mV]
const float xraxial = 1e9;       // [MOhm/cm]

// Allocate and fill host data
vector<short> vector_nrn_start_seg;
vector<string> vector_models;
vector<float> vector_Cm;
vector<float> vector_gnabar;
vector<float> vector_gkbar;
vector<float> vector_gl;
vector<float> vector_Ra;
vector<float> vector_diam;
vector<float> vector_length;
vector<float> vector_ena;
vector<float> vector_ek;
vector<float> vector_el;
vector<float> vector_gkrect;
vector<float> vector_gcaN;
vector<float> vector_gcaL;
vector<float> vector_gcak;
vector<float> vector_E_ex;
vector<float> vector_E_inh;
vector<float> vector_tau_exc;
vector<float> vector_tau_inh1;
vector<float> vector_tau_inh2;

vector <GroupMetadata> all_groups;

// form structs of neurons global ID and groups name
Group form_group(const string &group_name,
				 int nrns_in_group = neurons_in_group,
				 const string &model = INTER,
				 const int segs = 1) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = nrns_number;        // first ID in the group
	group.id_end = nrns_number + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group
	group.time = SIM_TIME_IN_STEPS;
	all_groups.emplace_back(group);

	float __Cm;
	float __gnabar;
	float __gkbar;
	float __gl;
	float __Ra;
	float __ena;
	float __ek;
	float __el;
	float __diam;
	float __dx;
	float __gkrect;
	float __gcaN;
	float __gcaL;
	float __gcak;
	float __e_ex;
	float __e_inh;
	float __tau_exc;
	float __tau_inh1;
	float __tau_inh2;

	for (int nrn = 0; nrn < nrns_in_group; nrn++) {
		if (model == INTER) {
			__Cm =  1; //random.gauss(1, 0.01);
			__gnabar = 0.1;
			__gkbar = 0.08;
			__gl = 0.002;
			__Ra = 100;
			__ena = 50;
			__ek = -90;
			__el = -70;
			__diam = 10; // random.randint(5, 15);
			__dx = __diam;
			__e_ex = 50;
			__e_inh = -80;
			__tau_exc = 0.35;
			__tau_inh1 = 0.5;
			__tau_inh2 = 3.5;
		} else if (model == MOTO) {
			__Cm = 2;
			__gnabar = 0.05;
			__gl = 0.002;
			__Ra = 200;
			__ena = 50;
			__ek = -80;
			__el = -70;
			__diam = 50; //random.randint(45, 55);
			__dx = __diam;
			__gkrect = 0.3;
			__gcaN = 0.05;
			__gcaL = 0.0001;
			__gcak = 0.3;
			__e_ex = 50;
			__e_inh = -80;
			__tau_exc = 0.3;
			__tau_inh1 = 1;
			__tau_inh2 = 1.5;
			if (__diam > 50) {
				__gnabar = 0.1;
				__gcaL = 0.001;
				__gl = 0.003;
				__gkrect = 0.2;
				__gcak = 0.2;
			}
		} else if (model == MUSCLE) {
			__Cm = 3.6;
			__gnabar = 0.15;
			__gkbar = 0.03;
			__gl = 0.0002;
			__Ra = 1.1;
			__ena = 55;
			__ek = -80;
			__el = -72;
			__diam = 40;
			__dx = 3000;
			__e_ex = 0;
			__e_inh = -80;
			__tau_exc = 0.3;
			__tau_inh1 = 1;
			__tau_inh2 = 1;
		} else if (model == GENERATOR) {

		} else {
			throw logic_error("Choose the model");
		}
		// common properties
		vector_Cm.push_back(__Cm);
		vector_gnabar.push_back(__gnabar);
		vector_gkbar.push_back(__gkbar);
		vector_gl.push_back(__gl);
		vector_el.push_back(__el);
		vector_ena.push_back(__ena);
		vector_ek.push_back(__ek);
		vector_Ra.push_back(__Ra);
		vector_diam.push_back(__diam);
		vector_length.push_back(__dx);
		vector_gkrect.push_back(__gkrect);
		vector_gcaN.push_back(__gcaN);
		vector_gcaL.push_back(__gcaL);
		vector_gcak.push_back(__gcak);
		vector_E_ex.push_back(__e_ex);
		vector_E_inh.push_back(__e_inh);
		vector_tau_exc.push_back(__tau_exc);
		vector_tau_inh1.push_back(__tau_inh1);
		vector_tau_inh2.push_back(__tau_inh2);
		//
		vector_nrn_start_seg.push_back(nrns_and_segs);
		nrns_and_segs += (segs + 2);
		vector_models.push_back(model);
	}

	nrns_number += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n",
		group_name.c_str(), nrns_number - nrns_in_group, nrns_number - 1, nrns_in_group);

	return group;
}

// copy data from host to device
template<typename type>
void memcpyHtD(type *gpu, type *host, unsigned int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

// copy data from device to host
template<typename type>
void memcpyDtH(type *host, type *gpu, unsigned int size) {
	cudaMemcpy(host, gpu, size * sizeof(type), cudaMemcpyDeviceToHost);
}

template<typename type>
type* init_gpu_arr(type *cpu_var, int size) {
	type *gpu_var;
	cudaMalloc(&gpu_var, size * sizeof(type));
	memcpyHtD<type>(gpu_var, cpu_var, size);
	return gpu_var;
}

template<typename type>
type *init_gpu_arr(vector<type> &vec) {
	type *gpu_var;
	cudaMalloc(&gpu_var, sizeof(type) * vec.size());
	memcpyHtD<type>(gpu_var, vec.data(), vec.size());
	return gpu_var;
}

template<typename type>
type* vec2arr(vector<type> &vec) {
	return vec.cpu_vector.data();
}
__device__
float Exp(float volt) {
	return (volt < -100)? 0 : exp(volt);
}

__device__
float alpham(float volt) {
	if (abs((volt + amB) / amC) < 1e-6)
		return amA * amC;
	return amA * (volt + amB) / (1 - Exp(-(volt + amB) / amC));
}

__device__
float betam(float volt) {
	if (abs((volt + bmB) / bmC) < 1e-6)
		return -bmA * bmC;
	return -bmA * (volt + bmB) / (1 - Exp((volt + bmB) / bmC));
}

__device__
float syn_current(Neurons &U, Parameters &P, int nrn, float voltage) {
	/**
	calculate synaptic current
	*/
	return U.g_exc[nrn] * (voltage - P.E_ex[nrn]) + (U.g_inh_B[nrn] - U.g_inh_A[nrn]) * (voltage - P.E_inh[nrn]);
}

__device__
float nrn_moto_current(States &S, Parameters &P, Neurons &U, int nrn, int nrn_seg_index, float voltage) {
	/**
	calculate channels current
	*/
	float iNa = P.gnabar[nrn] * pow(S.m[nrn_seg_index], 3) * S.h[nrn_seg_index] * (voltage - P.ena[nrn]);
	float iK = P.gkrect[nrn] * pow(S.n[nrn_seg_index], 4) * (voltage - P.ek[nrn]) +
               P.gcak[nrn] * pow(S.cai[nrn_seg_index], 2) / (pow(S.cai[nrn_seg_index], 2) + 0.014 * 0.014) * (voltage - P.ek[nrn]);
	float iL = P.gl[nrn] * (voltage - P.el[nrn]);
	float eCa = (1000 * R_const * 309.15 / (2 * F_const)) * log(ca0 / S.cai[nrn_seg_index]);
	S.I_Ca[nrn_seg_index] = P.gcaN[nrn] * pow(S.mc[nrn_seg_index], 2) * S.hc[nrn_seg_index] * (voltage - eCa) +
			                P.gcaL[nrn] * S.p[nrn_seg_index] * (voltage - eCa);
	return iNa + iK + iL + S.I_Ca[nrn_seg_index];
}

__device__
float nrn_fastchannel_current(States &S, Parameters &P, Neurons &U, int nrn, int nrn_seg_index, float voltage) {
	/**
	calculate channels current
	*/
	float iNa = P.gnabar[nrn] * pow(S.m[nrn_seg_index], 3) * S.h[nrn_seg_index] * (voltage - P.ena[nrn]);
	float iK = P.gkbar[nrn] * pow(S.n[nrn_seg_index], 4) * (voltage - P.ek[nrn]);
	float iL = P.gl[nrn] * (voltage - P.el[nrn]);
	return iNa + iK + iL;
}

__device__
void recalc_synaptic(States &S, Parameters &P, Neurons &U, int nrn) {
	/**
	updating conductance(summed) of neurons' post-synaptic conenctions
	*/
	// exc synaptic conductance
	if (U.g_exc[nrn] != 0) {
		U.g_exc[nrn] -= (1 - exp(-dt / P.tau_exc[nrn])) * U.g_exc[nrn];
		if (U.g_exc[nrn] < 1e-5)
			U.g_exc[nrn] = 0;
	}
	// inh1 synaptic conductance
	if (U.g_inh_A[nrn] != 0) {
		U.g_inh_A[nrn] -= (1 - exp(-dt / P.tau_inh1[nrn])) * U.g_inh_A[nrn];
		if (U.g_inh_A[nrn] < 1e-5)
			U.g_inh_A[nrn] = 0;
	}
	// inh2 synaptic conductance
	if (U.g_inh_B[nrn] != 0) {
		U.g_inh_B[nrn] -= (1 - exp(-dt / P.tau_inh2[nrn])) * U.g_inh_B[nrn];
		if (U.g_inh_B[nrn] < 1e-5)
			U.g_inh_B[nrn] = 0;
	}
}

__device__
void syn_initial(States &S, Parameters &P, Neurons &U, int nrn) {
	/**
	initialize tau(rise / decay time, ms) and factor(const) variables
	*/
	if (P.tau_inh1[nrn] / P.tau_inh2[nrn] > 0.9999)
		P.tau_inh1[nrn] = 0.9999 * P.tau_inh2[nrn];
	if (P.tau_inh1[nrn] / P.tau_inh2[nrn] < 1e-9)
		P.tau_inh1[nrn] = P.tau_inh2[nrn] * 1e-9;
	//
	float tp = (P.tau_inh1[nrn] * P.tau_inh2[nrn]) / (P.tau_inh2[nrn] - P.tau_inh1[nrn]) *
	           log(P.tau_inh2[nrn] / P.tau_inh1[nrn]);
	U.factor[nrn] = -exp(-tp / P.tau_inh1[nrn]) + exp(-tp / P.tau_inh2[nrn]);
	U.factor[nrn] = 1 / U.factor[nrn];
}

__device__
void nrn_inter_initial(States &S, Parameters &P, Neurons &U, int nrn_seg_index, float V) {
	/**
	initialize channels, based on cropped evaluate_fct function
	*/
	float V_mem = V - V_adj;
	//
	float a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	float b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S.m[nrn_seg_index] = a / (a + b);   // m_inf
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S.h[nrn_seg_index] = a / (a + b);   // h_inf
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S.n[nrn_seg_index] = a / (a + b);   // n_inf
}

__device__
void nrn_moto_initial(States &S, Parameters &P, Neurons &U, int nrn_seg_index, float V) {
	/** initialize channels, based on cropped evaluate_fct function */
	float a = alpham(V);
	S.m[nrn_seg_index] = a / (a + betam(V));                         // m_inf
	S.h[nrn_seg_index] = 1.0 / (1.0 + Exp((V + 65.0) / 7.0));   // h_inf
	S.p[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 55.8) / 3.7));  // p_inf
	S.n[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 38.0) / 15.0)); // n_inf
	S.mc[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 32.0) / 5.0)); // mc_inf
	S.hc[nrn_seg_index] = 1.0 / (1.0 + Exp((V + 50.0) / 5.0));  // hc_inf
	S.cai[nrn_seg_index] = 0.0001;
}

__device__
void nrn_muslce_initial(States &S, Parameters &P, Neurons &U, int nrn_seg_index, float V) {
	/**
	initialize channels, based on cropped evaluate_fct function
	*/
	float V_mem = V - V_adj;
	//
	float a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	float b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S.m[nrn_seg_index] = a / (a + b);   // m_inf
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S.h[nrn_seg_index] = a / (a + b);   // h_inf
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S.n[nrn_seg_index] = a / (a + b);   // n_inf
}

__device__
void recalc_inter_channels(States &S, Parameters &P, Neurons &U, int nrn_seg_index, float V) {
	/** calculate new states of channels (evaluate_fct) */
	// BREAKPOINT -> states -> evaluate_fct
	float V_mem = V - V_adj;
	//
	float a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	float b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	float tau = 1.0 / (a + b);
	float inf = a / (a + b);
	S.m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S.m[nrn_seg_index]);
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S.h[nrn_seg_index] += (1 - exp(-dt / tau)) * (inf - S.h[nrn_seg_index]);
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	// states
	S.n[nrn_seg_index] += (1 - exp(-dt / tau)) * (inf - S.n[nrn_seg_index]);
}

__device__
void nrn_rhs(States &S, Parameters &P, Neurons &U, int tid) {
	/**
	void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	calculate right hand side of
	cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	This is a common operation for fixed step, cvode, and daspk methods
	*/

}

__device__
void bksub(int tid) {
	/**
	void bksub(NrnThread* _nt)
	*/
}

__device__
void triang(int tid) {
	/**
	void triang(NrnThread* _nt)
	*/
}

__device__
void nrn_solve(States &S, Parameters &P, Neurons &U, int tid) {
	/**
	void nrn_solve(NrnThread* _nt)
	*/
	triang(tid);
	bksub(tid);

	S.Vm[tid] += P.Cm[tid] * P.ena[tid] + P.ek[tid];
}

__device__
void update(int tid) {
	/**
	void update(NrnThread* _nt)
	*/
}

__device__
void setup_tree_matrix(States &S, Parameters &P, Neurons &U, int tid) {
	nrn_rhs(S, P, U, tid);
	// simplified nrn_lhs(nrn)
	//	i1 = P.nrn_start_seg[nrn]
	//	i3 = P.nrn_start_seg[nrn + 1]
	//	S.NODE_D[i1:i3] += S.const_NODE_D[i1:i3]
}

__device__
void nrn_deliver_events(int tid) {
	/**
	void nrn_deliver_events(NrnThread* nt)
	*/
}

__device__
void nrn_fixed_step_lastpart(States &S, Parameters &P, Neurons &U, int tid) {
	/**
	void *nrn_fixed_step_lastpart(NrnThread *nth)
	*/
//	i1 = P.nrn_start_seg[nrn]
//	i3 = P.nrn_start_seg[nrn + 1]
//  update synapses' state
	recalc_synaptic(S, P, U, tid);
//  update neurons' segment state
//	if P.models[nrn] == INTER:
//	for nrn_seg in range(i1, i3):
//	recalc_inter_channels(nrn_seg, S.Vm[nrn_seg])
//	elif P.models[nrn] == MOTO:
//	for nrn_seg in range(i1, i3):
//	recalc_moto_channels(nrn_seg, S.Vm[nrn_seg])
//	elif P.models[nrn] == MUSCLE:
//	for nrn_seg in range(i1, i3):
//	recalc_muslce_channels(nrn_seg, S.Vm[nrn_seg])
//	else:
//	raise Exception("No model")
//  spike detection for
	nrn_deliver_events(tid);
}

__global__
void some_kernel(States *S, Parameters *P, Neurons *U, int neurons_number) {
	/// STRIDE neuron update
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < neurons_number; tid += blockDim.x * gridDim.x) {

		setup_tree_matrix(*S, *P, *U, tid);
		nrn_solve(*S, *P, *U, tid);
		update(tid);
		nrn_fixed_step_lastpart(*S, *P, *U, tid);
	}
}

void connect_fixed_outdegree(const Group &pre_neurons,
                             const Group &post_neurons,
                             float syn_delay,
                             float syn_weight,
                             int outdegree = 0,
                             bool no_distr = false) {
	// connect neurons with uniform distribution and normal distributon for syn delay and syn_weight
//	random_device r;
//	default_random_engine generator(r());
//	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
//	uniform_int_distribution<int> outdegree_num(30, 50);
//	normal_distribution<float> delay_distr_gen(syn_delay, syn_delay / 3);
//	normal_distribution<float> weight_distr_gen(syn_weight, syn_weight / 50);

//	if (outdegree == 0)
//		outdegree = outdegree_num(generator);

	int rand_post_id;
	float syn_delay_distr;
	float syn_weight_distr;

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
//			rand_post_id = id_distr(generator);
//			syn_delay_distr = delay_distr_gen(generator);

			if (syn_delay_distr < 0.1) {
				syn_delay_distr = 0.1;
			}
//			syn_weight_distr = weight_distr_gen(generator);

//			if (no_distr) {
//				all_synapses.emplace_back(pre_id, rand_post_id, syn_delay, syn_weight);
//			} else {
//				all_synapses.emplace_back(pre_id, rand_post_id, syn_delay_distr, syn_weight_distr);
//			}
		}
	}

//	printf("Connect %s to %s [fixed_outdegree] (1:%d). Total: %d W=%.2f, D=%.1f\n",
//	       pre_neurons.group_name.c_str(), post_neurons.group_name.c_str(),
//	       outdegree, pre_neurons.group_size * outdegree, syn_weight, syn_delay);
}

void init_network() {
	Group gen = form_group("gen", 1, GENERATOR, 1);
	Group OM1 = form_group("OM1", 50, INTER, 1);
	Group OM2 = form_group("OM2", 50, INTER, 1);
	Group OM3 = form_group("OM3", 50, INTER, 1);
	Group moto = form_group("moto", 50, MOTO, 1);
	Group muscle = form_group("muscle", 1, MUSCLE, 3);

//	conn_a2a(gen, OM1, delay=1, weight=1.5)

	connect_fixed_outdegree(OM1, OM2, 2, 1.85);
	connect_fixed_outdegree(OM2, OM1, 3, 1.85);
	connect_fixed_outdegree(OM2, OM3, 3, 0.00055);
	connect_fixed_outdegree(OM1, OM3, 3, 0.00005);
	connect_fixed_outdegree(OM3, OM2, 1, -4.5);
	connect_fixed_outdegree(OM3, OM1, 1, -4.5);
	connect_fixed_outdegree(OM2, moto, 2, 1.5);
	connect_fixed_outdegree(moto, muscle, 2, 15.5);

	vector<Group> groups = {OM1, OM2, OM3, moto, muscle};
//	save(groups)
//	P.nrn_start_seg.append(nrns_and_segs)
}

void simulate() {
	/**
	 *
	 */
	const int size = 10000;
	// init structs
	States *dev_S, *S = (States *)malloc(sizeof(States));
	Parameters *dev_P, *P = (Parameters *)malloc(sizeof(Parameters));
	Neurons *dev_U, *U = (Neurons *)malloc(sizeof(Neurons));

	init_network();

	/// finitialize()
	// init neurons
	for (int i = 0; i < size; i++) {
		vector_Cm.push_back(99);
		vector_ena.push_back(5);
		vector_ek.push_back(199);
	}
	/// GPU
	// init States CPU arrays
	auto *Vm = new float[size]();
	auto *n = new float[size]();
	auto *m = new float[size]();
	auto *h = new float[size]();
	auto *l = new float[size]();
	auto *s = new float[size]();
	auto *p = new float[size]();
	auto *hc = new float[size]();
	auto *mc = new float[size]();
	auto *cai = new float[size]();
	auto *I_Ca = new float[size]();
	auto *NODE_A = new float[size]();
	auto *NODE_B = new float[size]();
	auto *NODE_D = new float[size]();
	auto *const_NODE_D = new float[size]();
	auto *NODE_RHS = new float[size]();
	auto *NODE_RINV = new float[size]();
	auto *NODE_AREA = new float[size]();
	auto *has_spike = new bool[size]();
	auto *spike_on = new bool[size]();
	auto *g_exc = new float[size]();
	auto *g_inh_A = new float[size]();
	auto *g_inh_B = new float[size]();
	auto *factor = new float[size]();

	/// GPU
	// init Parameters (malloc + memcpy) GPU arrays based on CPU vectors
	short *gpu_nrn_start_seg = init_gpu_arr(vector_nrn_start_seg);
	string *gpu_models = init_gpu_arr(vector_models);
	auto *gpu_Cm = init_gpu_arr(vector_Cm);
	float *gpu_gnabar = init_gpu_arr(vector_gnabar);
	float *gpu_gkbar = init_gpu_arr(vector_gkbar);
	float *gpu_gl = init_gpu_arr(vector_gl);
	float *gpu_Ra = init_gpu_arr(vector_Ra);
	float *gpu_diam = init_gpu_arr(vector_diam);
	float *gpu_length = init_gpu_arr(vector_length);
	float *gpu_ena = init_gpu_arr(vector_ena);
	float *gpu_ek = init_gpu_arr(vector_ek);
	float *gpu_el = init_gpu_arr(vector_el);
	float *gpu_gkrect = init_gpu_arr(vector_gkrect);
	float *gpu_gcaN = init_gpu_arr(vector_gcaN);
	float *gpu_gcaL = init_gpu_arr(vector_gcaL);
	float *gpu_gcak = init_gpu_arr(vector_gcak);
	float *gpu_E_ex = init_gpu_arr(vector_E_ex);
	float *gpu_E_inh = init_gpu_arr(vector_E_inh);
	float *gpu_tau_exc = init_gpu_arr(vector_tau_exc);
	float *gpu_tau_inh1 = init_gpu_arr(vector_tau_inh1);
	float *gpu_tau_inh2 = init_gpu_arr(vector_tau_inh2);

	// init States GPU arrays based on CPU arrays
	auto *gpu_Vm = init_gpu_arr(Vm, size);
	auto *gpu_n = init_gpu_arr(n, size);
	auto *gpu_m = init_gpu_arr(m, size);
	auto *gpu_h = init_gpu_arr(h, size);
	auto *gpu_l = init_gpu_arr(l, size);
	auto *gpu_s = init_gpu_arr(s, size);
	auto *gpu_p = init_gpu_arr(p, size);
	auto *gpu_hc = init_gpu_arr(hc, size);
	auto *gpu_mc = init_gpu_arr(mc, size);
	auto *gpu_cai = init_gpu_arr(cai, size);
	auto *gpu_I_Ca = init_gpu_arr(I_Ca, size);
	auto *gpu_NODE_A = init_gpu_arr(NODE_A, size);
	auto *gpu_NODE_B = init_gpu_arr(NODE_B, size);
	auto *gpu_NODE_D = init_gpu_arr(NODE_D, size);
	auto *gpu_const_NODE_D = init_gpu_arr(const_NODE_D, size);
	auto *gpu_NODE_RHS = init_gpu_arr(NODE_RHS, size);
	auto *gpu_NODE_RINV = init_gpu_arr(NODE_RINV, size);
	auto *gpu_NODE_AREA = init_gpu_arr(NODE_AREA, size);
	auto *gpu_has_spike = init_gpu_arr(has_spike, size);
	auto *gpu_spike_on = init_gpu_arr(spike_on, size);
	auto *gpu_g_exc = init_gpu_arr(g_exc, size);
	auto *gpu_g_inh_A = init_gpu_arr(g_inh_A, size);
	auto *gpu_g_inh_B = init_gpu_arr(g_inh_B, size);
	auto *gpu_factor = init_gpu_arr(factor, size);

	// Point to device pointer in host struct
	// states
	S->Vm = gpu_Vm;
	S->n = gpu_n;
	S->m = gpu_m;
	S->h = gpu_h;
	S->l = gpu_l;
	S->s = gpu_s;
	S->p = gpu_p;
	S->hc = gpu_hc;
	S->mc = gpu_mc;
	S->cai = gpu_cai;
	S->I_Ca = gpu_I_Ca;
	S->NODE_A = gpu_NODE_A;
	S->NODE_B = gpu_NODE_B;
	S->NODE_D = gpu_NODE_D;
	S->const_NODE_D = gpu_const_NODE_D;
	S->NODE_RHS = gpu_NODE_RHS;
	S->NODE_RINV = gpu_NODE_RINV;
	S->NODE_AREA = gpu_NODE_AREA;
	// parameters
	P->nrn_start_seg = gpu_nrn_start_seg;
	P->models = gpu_models;
	P->Cm = gpu_Cm;
	P->gnabar = gpu_gnabar;
	P->gkbar = gpu_gkbar;
	P->gl = gpu_gl;
	P->Ra = gpu_Ra;
	P->diam = gpu_diam;
	P->length = gpu_length;
	P->ena = gpu_ena;
	P->ek = gpu_ek;
	P->el = gpu_el;
	P->gkrect = gpu_gkrect;
	P->gcaN = gpu_gcaN;
	P->gcaL = gpu_gcaL;
	P->gcak = gpu_gcak;
	P->E_ex = gpu_E_ex;
	P->E_inh = gpu_E_inh;
	P->tau_exc = gpu_tau_exc;
	P->tau_inh1 = gpu_tau_inh1;
	P->tau_inh2 = gpu_tau_inh2;
	// Neurons
	U->has_spike = gpu_has_spike;
	U->spike_on = gpu_spike_on;
	U->g_exc = gpu_g_exc;
	U->g_inh_A = gpu_g_inh_A;
	U->g_inh_B = gpu_g_inh_B;
	U->factor = gpu_factor;

	// allocate States struct to the device
	cudaMalloc((States **) &dev_S, sizeof(States));
	cudaMemcpy(dev_S, S, sizeof(States), cudaMemcpyHostToDevice);
	// allocate Parameters struct to the device
	cudaMalloc((Parameters **) &dev_P, sizeof(Parameters));
	cudaMemcpy(dev_P, P, sizeof(Parameters), cudaMemcpyHostToDevice);
	// allocate Neurons struct to the device
	cudaMalloc((Neurons **) &dev_U, sizeof(Neurons));
	cudaMemcpy(dev_U, U, sizeof(Neurons), cudaMemcpyHostToDevice);

	// call kernel
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		some_kernel<<<10000 / 256 + 1, 256>>>(dev_S, dev_P, dev_U, size); // block size need to be a multiply of 256
	}

	CHECK(cudaDeviceSynchronize());

	// Copy result to host:
	CHECK(cudaMemcpy(Vm, gpu_Vm, size * sizeof(*Vm), cudaMemcpyDeviceToHost));

	// Print some result
	cout << Vm[size-10] << std::endl;

	CHECK(cudaFree(gpu_Vm));
}

int main(int argc, char **argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	//
	CHECK(cudaSetDevice(dev));
	//
	simulate();
	// reset device
	CHECK(cudaDeviceReset());
}
