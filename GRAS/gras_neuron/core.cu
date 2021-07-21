/**
See the topology https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
Based on the NEURON repository.
*/
//#define LOG
#include <omp.h>
#include <assert.h>
#include <random>
#include <vector>
#include <string>
#include "structs.h"
#include <stdexcept>
#include <curand_kernel.h>
// for file writing
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
#define PI 3.141592654f

using namespace std;

random_device r;
default_random_engine rand_gen(r());

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("!!! %s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

const double dt = 0.025;      // [ms] simulation step
const int cv_fr = 200;      // frequency of CV
int step_number;  // [step] number of full cycle steps
const int ees_fr = 40;      // frequency of EES
const bool EXTRACELLULAR = false;

const char layers = 5;      // number of OM layers (5 is default)
unsigned int one_step_time;
int skin_time;   // duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
int slices_extensor;
unsigned int slice_time = 25;
double E_coef;
int slices_flexor; // flexor duration (125 or 175 ms for 4pedal)
double cv_coef;
bool str_flag = false;

unsigned int sim_time;
unsigned int SIM_TIME_IN_STEPS;

unsigned int NRNS_NUMBER = 0;     // [id] global neuron id = number of neurons
unsigned int NRNS_AND_SEGS = 0;   // [id] global neuron+segs id = number of neurons with segments
const int neurons_in_group = 50;  // number of neurons in a group
const int neurons_in_ip = 196;    // number of neurons in a group

// common neuron constants
// normal

const double V_th = -40;          // [mV] voltage threshold
const double V_adj = -63;         // [mV] adjust voltage for -55 threshold
// moto neuron constants
const double amA = 0.4;           // const ??? todo
const double amB = 66;            // const ??? todo
const double amC = 5;             // const ??? todo
const double bmA = 0.4;           // const ??? todo
const double bmB = 32;            // const ??? todo
const double bmC = 5;             // const ??? todo
const double ca0 = 2;             // initial calcium concentration
const double R_const = 8.314472;  // [k-mole] or [joule/degC] const
const double F_const = 96485.34;  // [faraday] or [kilocoulombs] const
// muscle fiber constants
// const double g_kno = 0.01;     // [S/cm2] conductance of the todo
// const double g_kir = 0.03;     // [S/cm2] conductance of the Inwardly Rectifying Potassium K+ (Kir) channel
// Boltzman steady state curve
const double vhalfl = -98.92;     // [mV] inactivation half-potential
const double kl = 10.89;          // [mV] Stegen et al. 2012
// tau_infty
const double vhalft = 67.0828;    // [mV] fitted //100 uM sens curr 350a, Stegen et al. 2012
const double at = 0.00610779;     // [/ ms] Stegen et al. 2012
const double bt = 0.0817741;      // [/ ms] Note: typo in Stegen et al. 2012
// temperature dependence
const double q10 = 1;             // temperature scaling (sensitivity)
const double celsius = 36;        // [degC] temperature of the cell
// i_membrane [mA/cm2]
const int nlayer = 2;
const double e_extracellular = 0; // [mV]
const double xraxial = 1e9;       // [MOhm/cm]

// neuron parameters
vector<unsigned int> vector_nrn_start_seg;
vector<char> vector_models;
vector<double> vector_Cm, vector_gnabar, vector_gkbar, vector_gl, vector_Ra, vector_diam, vector_length, vector_ena,
               vector_ek, vector_el, vector_gkrect, vector_gcaN, vector_gcaL, vector_gcak;
// synaptic parameters
vector<double> vector_E_ex, vector_E_inh, vector_tau_exc, vector_tau_inh1, vector_tau_inh2;
// synapses varaibels
vector<int> vector_syn_pre_nrn, vector_syn_post_nrn, vector_syn_delay, vector_syn_delay_timer;
vector<double> vector_syn_weight;
// results vector
vector <GroupMetadata> saving_groups;
// for debugging
vector <Group> all_groups;
// generators
vector<unsigned int> vec_time_end, vec_nrn_id, vec_freq_in_steps, vec_spike_each_step;


double *bimodal_distr_for_moto_neurons(const unsigned int nrnnumber) {
	int standby_percent = 70;
	double diameter_active = 27.0;
	double diameter_standby = 57.0;

	double* nrn_diameter = new double [nrnnumber];
	random_device r1;
	default_random_engine generator1(r1());
	normal_distribution<double> d_active(diameter_active, 3);
	normal_distribution<double> d_standby(diameter_standby, 6);

	int standby_size = (int) (nrnnumber * standby_percent / 100);
	int active_size = nrnnumber - standby_size;

	for (int i = 0; i < active_size; i++)
		nrn_diameter[i] = d_active(generator1);
	for (int i = active_size; i < nrnnumber; i++)
		nrn_diameter[i] = d_standby(generator1);
	return nrn_diameter;
}

// form structs of neurons global ID and groups name
Group form_group(const string &group_name,
                 int nrns_in_group = 50,
                 const char model = INTER,
                 const int segs = 1) {
	/**
	 *
	 */
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = NRNS_NUMBER;      // first ID in the group
	group.id_end = NRNS_NUMBER + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group
	group.model = model;

	double Cm, gnabar, gkbar, gl, Ra, ena, ek, el, diam, dx, gkrect, gcaN, gcaL, gcak, e_ex, e_inh, tau_exc, tau_inh1, tau_inh2;
	uniform_real_distribution<double> Cm_distr(0.3, 2.5);
	uniform_real_distribution<double> Cm_distr_muscle(2.5, 4.0);
	uniform_real_distribution<double> length_distr_muscle(2500, 3500);
	normal_distribution<double> moto_Cm_distr(2, 0.5);
	uniform_int_distribution<int> inter_diam_distr(5, 15);
	uniform_real_distribution<double> afferent_diam_distr(15, 35);

	uniform_real_distribution<double> gl_distr_MUSCLE(0.0005, 0.001); // 8, 12
	uniform_real_distribution<double> tau_exc_distr_MUSCLE(0.33, 0.35);

	double* diameters; //
	if (model == MOTO)
		diameters = bimodal_distr_for_moto_neurons(nrns_in_group);

	for (int nrn = 0; nrn < nrns_in_group; nrn++) {
		if (model == INTER) {
			Cm = Cm_distr(rand_gen);
			gnabar = 0.1;
			gkbar = 0.08;
			gl = 0.002;
			Ra = 100.0;
			ena = 50.0;
			ek = -77.0;
			el = -70.0;
			diam = inter_diam_distr(rand_gen);
			dx = diam;
			e_ex = 50;
			e_inh = -80;
			tau_exc = 0.35;
			tau_inh1 = 0.5;
			tau_inh2 = 3.5;
		}
		else if (model == AFFERENTS) {
			Cm = 2;
			gnabar = 0.5;
			gkbar = 0.04;
			gl = 0.002;
			Ra = 200.0;
			ena = 50.0;
			ek = -90.0;
			el = -70.0;
			diam = afferent_diam_distr(rand_gen); // 10
			dx = diam;
			e_ex = 50;
			e_inh = -80;
			tau_exc = 0.35;
			tau_inh1 = 0.5;
			tau_inh2 = 3.5;
		}
		else if (model == MOTO) {
			Cm = moto_Cm_distr(rand_gen);
			gnabar = 0.05;
			gl = 0.002;
			Ra = 200.0;
			ena = 50.0;
			ek = -80.0;
			el = -70.0;
			diam = diameters[nrn];
			dx = diam;
			gkrect = 0.3;
			gcaN = 0.05;
			gcaL = 0.0001;
			gcak = 0.3;
			e_ex = 50.0;
			e_inh = -80.0;
			tau_exc = 0.3;
			tau_inh1 = 1.0;
			tau_inh2 = 1.5;
			if (diam > 50) {
				gnabar = 0.1;
				gcaL = 0.001;
				gl = 0.003;
				gkrect = 0.2;
				gcak = 0.2;
			}
		} else if (model == MUSCLE) {
			Cm = Cm_distr_muscle(rand_gen);
			gnabar = 0.03;
			gkbar = 0.06;
//			gl = 0.001;
			gl = gl_distr_MUSCLE(rand_gen);
			Ra = 1.1;
			ena = 55.0;
			ek = -90.0;
			el = -70.0;
			diam = 40.0;
			dx = length_distr_muscle(rand_gen);
			e_ex = 0.0;
			e_inh = -80.0;
			tau_exc = 0.35;
//			tau_exc = tau_exc_distr_MUSCLE(rand_gen);
			tau_inh1 = 1.0;
			tau_inh2 = 1.0;
		} else if (model == GENERATOR) {
			// nothing
		} else {
			throw logic_error("Choose the model");
		}
		// common properties
		vector_Cm.push_back(Cm);
		vector_gnabar.push_back(gnabar);
		vector_gkbar.push_back(gkbar);
		vector_gl.push_back(gl);
		vector_el.push_back(el);
		vector_ena.push_back(ena);
		vector_ek.push_back(ek);
		vector_Ra.push_back(Ra);
		vector_diam.push_back(diam);
		vector_length.push_back(dx);
		vector_gkrect.push_back(gkrect);
		vector_gcaN.push_back(gcaN);
		vector_gcaL.push_back(gcaL);
		vector_gcak.push_back(gcak);
		vector_E_ex.push_back(e_ex);
		vector_E_inh.push_back(e_inh);
		vector_tau_exc.push_back(tau_exc);
		vector_tau_inh1.push_back(tau_inh1);
		vector_tau_inh2.push_back(tau_inh2);
		//
		vector_nrn_start_seg.push_back(NRNS_AND_SEGS);
		NRNS_AND_SEGS += (segs + 2);
		vector_models.push_back(model);
		// void (*foo)(int);
		// foo = &my_int_func;
	}

	NRNS_NUMBER += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n",
	       group_name.c_str(), NRNS_NUMBER - nrns_in_group, NRNS_NUMBER - 1, nrns_in_group);

	// for debugging
	all_groups.push_back(group);

	return group;
}

__host__
unsigned int ms_to_step(double ms) { return (unsigned int) (ms / dt); }

__host__
double step_to_ms(int step) { return step * dt; }

// copy data from host to device
template<typename type>
void memcpyHtD(type *host, type *gpu, unsigned int size) {
	HANDLE_ERROR(cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice));
}

// copy data from device to host
template<typename type>
void memcpyDtH(type *gpu, type *host, unsigned int size) {
	HANDLE_ERROR(cudaMemcpy(host, gpu, size * sizeof(type), cudaMemcpyDeviceToHost));
}

// init GPU array and copy data from the CPU array
template<typename type>
type* init_gpu_arr(type *cpu_var, unsigned int size = NRNS_AND_SEGS) {
	type *gpu_var;
	HANDLE_ERROR(cudaMalloc(&gpu_var, size * sizeof(type)));
	memcpyHtD<type>(cpu_var, gpu_var, size);
	return gpu_var;
}

// init GPU array and copy data from the CPU vector
template<typename type>
type *init_gpu_arr(vector<type> &vec) {
	type *gpu_var;
	HANDLE_ERROR(cudaMalloc(&gpu_var, sizeof(type) * vec.size()));
	memcpyHtD<type>(vec.data(), gpu_var, vec.size());
	return gpu_var;
}

void add_generator(Group &group, double start, double end, double freq) {
	vec_nrn_id.push_back(group.id_start);
	vec_time_end.push_back(ms_to_step(end));
	vec_freq_in_steps.push_back(ms_to_step(1000 / freq));
	vec_spike_each_step.push_back(ms_to_step(start));
	printf("start %d end %d freq %d\n", ms_to_step(start), ms_to_step(end), ms_to_step(1000 / freq));
}

// convert vector to the array
template<typename type>
type* vec2arr(vector<type> &vec) {
	return vec.cpu_vector.data();
}

__device__
double Exp(double volt) {
	return (volt < -100)? 0 : exp(volt);
}

__device__
double alpham(double volt) {
	if (abs((volt + amB) / amC) < 1e-6)
		return amA * amC;
	return amA * (volt + amB) / (1.0 - Exp(-(volt + amB) / amC));
}

__device__
double betam(double volt) {
	if (abs((volt + bmB) / bmC) < 1e-6)
		return -bmA * bmC;
	return -bmA * (volt + bmB) / (1.0 - Exp((volt + bmB) / bmC));
}

__device__
double syn_current(Neurons* N, const Parameters* P, int nrn, double voltage) {
	/**
	 * calculate synaptic current
	 */
	return N->g_exc[nrn] * (voltage - P->E_ex[nrn]) + (N->g_inh_B[nrn] - N->g_inh_A[nrn]) * (voltage - P->E_inh[nrn]);
}

__device__
double nrn_moto_current(States* S, const Parameters* P, Neurons* N, int nrn, int nrn_seg_index, double voltage) {
	/**
	 * calculate channels current
	 */
	double iNa = P->gnabar[nrn] * pow(S->m[nrn_seg_index], 3) * S->h[nrn_seg_index] * (voltage - P->ena[nrn]);
	double iK = P->gkrect[nrn] * pow(S->n[nrn_seg_index], 4) * (voltage - P->ek[nrn]) +
	            P->gcak[nrn] * pow(S->cai[nrn_seg_index], 2) / (pow(S->cai[nrn_seg_index], 2) + 0.014 * 0.014) * (voltage - P->ek[nrn]);
	double iL = P->gl[nrn] * (voltage - P->el[nrn]);
	double E_Ca = (1000.0 * R_const * 309.15 / (2.0 * F_const)) * log(ca0 / S->cai[nrn_seg_index]);
	S->I_Ca[nrn_seg_index] = P->gcaN[nrn] * S->mc[nrn_seg_index] * S->mc[nrn_seg_index] * S->hc[nrn_seg_index] * (voltage - E_Ca) +
	                         P->gcaL[nrn] * S->p[nrn_seg_index] * (voltage - E_Ca);
	return iNa + iK + iL + S->I_Ca[nrn_seg_index];
}

__device__
double nrn_fastchannel_current(States* S, const Parameters* P, Neurons* N, int nrn, int nrn_seg_index, double voltage) {
	/**
	 * calculate channels current
	 */
	double iNa = P->gnabar[nrn] * pow(S->m[nrn_seg_index], 3) * S->h[nrn_seg_index] * (voltage - P->ena[nrn]);
	double iK = P->gkbar[nrn] * pow(S->n[nrn_seg_index], 4) * (voltage - P->ek[nrn]);
	double iL = P->gl[nrn] * (voltage - P->el[nrn]);
	return iNa + iK + iL;
}

__device__
void recalc_synaptic(States* S, const Parameters* P, Neurons* N, int nrn) {
	/**
	 * updating conductance(summed) of neurons' post-synaptic conenctions
	 */
	// exc synaptic conductance
	if (N->g_exc[nrn] != 0.0) {
		N->g_exc[nrn] -= (1.0 - exp(-dt / P->tau_exc[nrn])) * N->g_exc[nrn];
		if (N->g_exc[nrn] < 1e-5) {
			N->g_exc[nrn] = 0.0;
		}
	}
	// inh1 synaptic conductance
	if (N->g_inh_A[nrn] != 0.0) {
		N->g_inh_A[nrn] -= (1.0 - exp(-dt / P->tau_inh1[nrn])) * N->g_inh_A[nrn];
		if (N->g_inh_A[nrn] < 1e-5) {
			N->g_inh_A[nrn] = 0.0;
		}
	}
	// inh2 synaptic conductance
	if (N->g_inh_B[nrn] != 0.0) {
		N->g_inh_B[nrn] -= (1.0 - exp(-dt / P->tau_inh2[nrn])) * N->g_inh_B[nrn];
		if (N->g_inh_B[nrn] < 1e-5)
			N->g_inh_B[nrn] = 0.0;
	}
}

__device__
void syn_initial(States* S, const Parameters* P, Neurons* N, int nrn) {
	/**
	 * initialize tau(rise / decay time, ms) and factor(const) variables
	 */
	if (P->tau_inh1[nrn] / P->tau_inh2[nrn] > 0.9999)
		P->tau_inh1[nrn] = 0.9999 * P->tau_inh2[nrn];
	if (P->tau_inh1[nrn] / P->tau_inh2[nrn] < 1e-9)
		P->tau_inh1[nrn] = P->tau_inh2[nrn] * 1e-9;
	//
	double tp = (P->tau_inh1[nrn] * P->tau_inh2[nrn]) / (P->tau_inh2[nrn] - P->tau_inh1[nrn]) *
	            log(P->tau_inh2[nrn] / P->tau_inh1[nrn]);
	N->factor[nrn] = -exp(-tp / P->tau_inh1[nrn]) + exp(-tp / P->tau_inh2[nrn]);
	N->factor[nrn] = 1.0 / N->factor[nrn];
}

__device__
void nrn_inter_initial(States* S, const Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * initialize channels, based on cropped evaluate_fct function
	 */
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S->m[nrn_seg_index] = a / (a + b);   // m_inf
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S->h[nrn_seg_index] = a / (a + b);   // h_inf
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S->n[nrn_seg_index] = a / (a + b);   // n_inf
}

__device__
void nrn_moto_initial(States* S, const Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * initialize channels, based on cropped evaluate_fct function
	 */
	double a = alpham(V);
	S->m[nrn_seg_index] = a / (a + betam(V));                         // m_inf
	S->h[nrn_seg_index] = 1.0 / (1.0 + Exp((V + 65.0) / 7.0));   // h_inf
	S->p[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 55.8) / 3.7));  // p_inf
	S->n[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 38.0) / 15.0)); // n_inf
	S->mc[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 32.0) / 5.0)); // mc_inf
	S->hc[nrn_seg_index] = 1.0 / (1.0 + Exp((V + 50.0) / 5.0));  // hc_inf
	S->cai[nrn_seg_index] = 0.0001;
}

__device__
void nrn_muslce_initial(States* S, const Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * initialize channels, based on cropped evaluate_fct function
	 */
	double V_mem = V - V_adj;
	// m_inf
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S->m[nrn_seg_index] = a / (a + b);
	// h_inf
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S->h[nrn_seg_index] = a / (a + b);
	// n_inf
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S->n[nrn_seg_index] = a / (a + b);
}

__device__
void recalc_inter_channels(States* S, const Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * calculate new states of channels (evaluate_fct)
	 */
	// BREAKPOINT -> states -> evaluate_fct
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	double tau = 1.0 / (a + b);
	double inf = a / (a + b);
	S->m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->m[nrn_seg_index]);
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S->h[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->h[nrn_seg_index]);
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	// states
	S->n[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->n[nrn_seg_index]);
}

__device__
void recalc_moto_channels(States* S, const Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * calculate new states of channels (evaluate_fct)
	 */
	// BREAKPOINT -> states -> evaluate_fct
	double a = alpham(V);
	double b = betam(V);
	// m
	double tau = 1.0 / (a + b);
	double inf = a / (a + b);
	S->m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->m[nrn_seg_index]);
	// h
	tau = 30.0 / (Exp((V + 60.0) / 15.0) + Exp(-(V + 60.0) / 16.0));
	inf = 1.0 / (1 + Exp((V + 65.0) / 7.0));
	S->h[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->h[nrn_seg_index]);
	// DELAYED RECTIFIER POTASSIUM
	tau = 5.0 / (Exp((V + 50.0) / 40.0) + Exp(-(V + 50.0) / 50.0));
	inf = 1.0 / (1.0 + Exp(-(V + 38.0) / 15.0));
	S->n[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->n[nrn_seg_index]);
	// CALCIUM DYNAMICS N-type
	double mc_inf = 1.0 / (1.0 + Exp(-(V + 32.0) / 5.0));
	double hc_inf = 1.0 / (1.0 + Exp((V + 50.0) / 5.0));
	// CALCIUM DYNAMICS L-type
	tau = 400.0;
	inf = 1.0 / (1.0 + Exp(-(V + 55.8) / 3.7));
	S->p[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->p[nrn_seg_index]);
	// states
	S->mc[nrn_seg_index] += (1.0 - exp(-dt / 15.0)) * (mc_inf - S->mc[nrn_seg_index]);     // tau_mc = 15
	S->hc[nrn_seg_index] += (1.0 - exp(-dt / 50.0)) * (hc_inf - S->hc[nrn_seg_index]);     // tau_hc = 50
	S->cai[nrn_seg_index] += (1.0 - exp(-dt * 0.04)) * (-0.01 * S->I_Ca[nrn_seg_index] / 0.04 - S->cai[nrn_seg_index]);
}

__device__
void recalc_muslce_channels(States* S, const Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * calculate new states of channels (evaluate_fct)
	 */
	// BREAKPOINT -> states -> evaluate_fct
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	double tau = 1.0 / (a + b);
	double inf = a / (a + b);
	S->m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->m[nrn_seg_index]);
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S->h[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->h[nrn_seg_index]);
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S->n[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->n[nrn_seg_index]);
//	//
//	double qt = pow(q10, (celsius - 33.0) / 10.0);
//	double linf = 1.0 / (1.0 + exp((V - vhalfl) / kl)); // l_steadystate
//	double taul = 1.0 / (qt * (at * exp(-V / vhalft) + bt * exp(V / vhalft)));
//	double alpha = 0.3 / (1.0 + exp((V + 43.0) / -5.0));
//	double beta = 0.03 / (1.0 + exp((V + 80.0) / -1.0));
//	double stau = 1.0 / (alpha + beta);
//	double sinf = alpha / (alpha + beta);
//	// states
//	S->l[nrn_seg_index] += (1.0 - exp(-dt / taul)) * (linf - S->l[nrn_seg_index]);
//	S->s[nrn_seg_index] += (1.0 - exp(-dt / stau)) * (sinf - S->s[nrn_seg_index]);
}

__device__
void nrn_rhs_ext(States* S, const Parameters* P, Neurons* N, int i1, int i3) {
	/**
	 * void nrn_rhs_ext(NrnThread* _nt)
	 */
	int size = S->ext_size;
	const double xg[5] = {0, 1e9, 1e9, 1e9, 0};
	//
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		for (int layer = 0; layer < nlayer; ++layer) {  // zeroed at nrn_rhs before nrn_rhs_ext
			S->EXT_RHS[nrn_seg + layer * size] = 0;
		}
		S->EXT_RHS[nrn_seg + 0 * size] -= S->NODE_RHS[nrn_seg];
	}
#ifdef LOG
	printf("nrn_rhs_ext::EXT RHS 0 : "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_RHS[ii]); printf("\n");
	printf("nrn_rhs_ext::EXT RHS 1 : "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_RHS[ii + size]); printf("\n");
#endif
	double x, dv;
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
#ifdef LOG
		printf("V: %g, %g\n", S->EXT_V[nrn_seg], S->EXT_V[nrn_seg + size]);
#endif
		for (int layer = 0; layer < nlayer; ++layer) {
			dv = S->EXT_V[nrn_seg - 1 + layer * size] - S->EXT_V[nrn_seg + layer * size];
			S->EXT_RHS[nrn_seg + layer * size] -= S->EXT_B[nrn_seg + layer * size] * dv;
			S->EXT_RHS[nrn_seg - 1 + layer * size] += S->EXT_A[nrn_seg + layer * size] * dv;
		}
		int layer = nlayer - 1;
		S->EXT_RHS[nrn_seg + layer * size] -= xg[nrn_seg - i1] * (S->EXT_RHS[nrn_seg + layer * size] - e_extracellular);
		for (--layer; layer >= 0; --layer) { /* between j and j+1 layer */
			x = xg[nrn_seg - i1] * (S->EXT_V[nrn_seg + layer * size] - S->EXT_V[nrn_seg + (layer + 1) * size]);
			S->EXT_RHS[nrn_seg + layer * size] -= x;
			S->EXT_RHS[nrn_seg + (layer + 1) * size] += x;
		}
	}
#ifdef LOG
	printf("nrn_rhs_ext::EXT RHS END 0 : "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_RHS[ii]); printf("\n");
	printf("nrn_rhs_ext::EXT RHS END 1 : "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_RHS[ii + size]); printf("\n");
#endif
}

__device__
void nrn_setup_ext(States* S, const Parameters* P, Neurons* N, int i1, int i3) {
	/**
	 * void nrn_setup_ext(NrnThread* _nt)
	 */
	double cj = 1 / dt;
	double cfac = 0.001 * cj;
	int size = S->ext_size;
	const double xg[5] = {0, 1e9, 1e9, 1e9, 0};
	const double xc[5] = {0, 0, 0, 0, 0};
	// todo find the place where it is zeroed
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg)
		for (int layer = 0; layer < nlayer; ++layer)
			S->EXT_D[nrn_seg + layer * size] = 0;
	// d contains all the membrane conductances (and capacitance)
	// i.e. (cm/dt + di/dvm - dis/dvi)*[dvi] and (dis/dvi)*[dvx]
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		// nde->_d only has -ELECTRODE_CURRENT contribution
		S->EXT_D[nrn_seg + 0 * size] += S->NODE_D[nrn_seg];
	}
	// NEURON D 0 = [0 0.1442 0.1442 0.1442 0 ] [0 0 0 0 0]
	// GRAS [0 0.1442 0.1442 0.1442 0 ] [0 0 0 0 0]
	// series resistance, capacitance, and axial terms
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		// series resistance and capacitance to ground
		int layer = 0;
		while (1) {
			double mfac = xg[nrn_seg - i1] + xc[nrn_seg - i1] * cfac;
			S->EXT_D[nrn_seg + layer * size] += mfac;
			layer += 1;
			if (layer == nlayer)
				break;
			S->EXT_D[nrn_seg + layer * size] += mfac;
		}
		// axial connections
		for (layer = 0; layer < nlayer; ++layer) {
			S->EXT_D[nrn_seg + layer * size] -= S->EXT_B[nrn_seg + layer * size];
			S->EXT_D[nrn_seg - 1 + layer * size] -= S->EXT_A[nrn_seg + layer * size];
		}
	}
	// NEURON D[0] = [2e-08 1e+09 1e+09 1e+09 2e-08 ] GRAS [2e-08 1e+09 1e+09 1e+09 2e-08]
	// NEURON D[1] = [2e-08 2e+09 2e+09 2e+09 2e-08 ] GRAS [2e-08 2e+09 2e+09 2e+09 2e-08]
}

__device__
void nrn_update_2d(States* S, const Parameters* P, Neurons* N, int i1, int i3) {
	/**
	 * void nrn_update_2d(NrnThread* nt)
	 * update has already been called so modify nd->v based on dvi we only need to
	 * update extracellular nodes and base the corresponding nd->v on dvm (dvm = dvi - dvx)
	 */
	// final voltage updating
	int size = S->ext_size;
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg)
		for (int layer = 0; layer < nlayer; ++layer)
			S->EXT_V[nrn_seg + layer * size] += S->EXT_RHS[nrn_seg + layer * size];
}

__device__
void nrn_rhs(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	 * calculate right hand side of
	 * cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	 * cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	 * This is a common operation for fixed step, cvode, and daspk methods
	 */
	// init _rhs and _lhs (NODE_D) as zero
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_RHS[nrn_seg] = 0;
		// replace the process: init by 0, add Cm*frac, add A and B
	}

	// update MOD rhs, CAPS has no current [CAP MOD CAP]!
	// int center_segment = i1 + ((P->models[nrn] == MUSCLE)? 2 : 1);
	// update segments except CAPs
	double V, _rhs;
	for (int nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
		V = S->Vm[nrn_seg];
		// SYNAPTIC update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		_rhs = syn_current(N, P, nrn, V);
		_rhs *= 1.e2 / S->NODE_AREA[nrn_seg];
		S->NODE_RHS[nrn_seg] -= _rhs;

		// NEURON update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
			// muscle and inter has the same fast_channel function
			_rhs = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V);
		} else if (P->models[nrn] == MOTO) {
			_rhs = nrn_moto_current(S, P, N, nrn, nrn_seg, V);
		} else if (P->models[nrn] == MUSCLE) {
			// muscle and inter has the same fast_channel function
			_rhs = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V);
		} else {
			printf("\nERROR\n");
		}
		// save data like in NEURON (after .mod nrn_cur)
		S->NODE_RHS[nrn_seg] -= _rhs;
		// note that CAP has no jacob
	} // end FOR segments
	if (EXTRACELLULAR) {
		// Cannot have any axial terms yet so that i(vm) can be calculated from
		// i(vm)+is(vi) and is(vi) which are stored in rhs vector.
		nrn_rhs_ext(S, P, N, i1, i3);
		// nrn_rhs_ext has also computed the the internal axial current for those
		// nodes containing the extracellular mechanism
	}
	double dv;
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		dv = S->Vm[nrn_seg - 1] - S->Vm[nrn_seg];
		// our connection coefficients are negative so
		S->NODE_RHS[nrn_seg] -= S->NODE_B[nrn_seg] * dv;
		S->NODE_RHS[nrn_seg - 1] += S->NODE_A[nrn_seg] * dv;
	}
#ifdef  LOG
	printf("RHS EXRTA 0 : "); for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) printf("%g\t", S->EXT_RHS[nrn_seg]); printf("\n");
	printf("RHS EXRTA 1 : "); for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) printf("%g\t", S->EXT_RHS[nrn_seg + S->ext_size]); printf("\n");
#endif
}

__device__
void nrn_lhs(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/** calculate left hand side of
	 * cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	 * cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	 * with a matrix so that the solution is of the form [dvm+dvx,dvx] on the right hand side after solving.
	 */
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_D[nrn_seg] = 0;
	}
	// update MOD rhs, CAPS has no current [CAP MOD CAP]!
	// int center_segment = i1 + ((P->models[nrn] == MUSCLE)? 2 : 1);
	// update segments except CAPs
	double V, _g, _rhs;
	for (int nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
		V = S->Vm[nrn_seg];
		// SYNAPTIC update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		_g = syn_current(N, P, nrn, V + 0.001);
		_rhs = syn_current(N, P, nrn, V);
		_g = (_g - _rhs) / 0.001;
		_g *= 1.e2 / S->NODE_AREA[nrn_seg];
		_rhs *= 1.e2 / S->NODE_AREA[nrn_seg];
		// static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
		S->NODE_D[nrn_seg] += _g;

		// void nrn_cap_jacob(NrnThread* _nt, Memb_list* ml) {
		S->NODE_D[nrn_seg] += S->const_NODE_D[nrn_seg];
		// activsynapse_lhs()
		// activclamp_lhs()

		// NEURON update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
			// muscle and inter has the same fast_channel function
			_g = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V + 0.001);
			_rhs = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V);
		} else if (P->models[nrn] == MOTO) {
			_g = nrn_moto_current(S, P, N, nrn, nrn_seg, V + 0.001);
			_rhs = nrn_moto_current(S, P, N, nrn, nrn_seg, V);
		} else if (P->models[nrn] == MUSCLE) {
			// muscle and inter has the same fast_channel function
			_g = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V + 0.001);
			_rhs = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V);
		} else {
			assert(false);
		}
		// save data like in NEURON (after .mod nrn_cur)
		_g = (_g - _rhs) / 0.001;
		// note that CAP has no jacob
		S->NODE_D[nrn_seg] += _g;
	} // end FOR segments

	if (EXTRACELLULAR) {
		nrn_setup_ext(S, P, N, i1, i3);
	}
	// activstim_rhs()
	// activclamp_rhs()

	// at this point d contains all the membrane conductances
	// now add the axial currents
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_D[nrn_seg] -= S->NODE_B[nrn_seg];
		S->NODE_D[nrn_seg - 1] -= S->NODE_A[nrn_seg];
	}
#ifdef LOG
	printf("NODED axial:\t"); for (int i = i1; i < i3; ++i) printf("%g\t", S->NODE_D[i]); printf("\n");
#endif
}

__device__
void bksub(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void bksub(NrnThread* _nt)
	 */
	// intracellular
	S->NODE_RHS[i1] /= S->NODE_D[i1];
	//
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_RHS[nrn_seg] -= S->NODE_B[nrn_seg] * S->NODE_RHS[nrn_seg - 1];
		S->NODE_RHS[nrn_seg] /= S->NODE_D[nrn_seg];
	}
	// extracellular
//	if (EXTRACELLULAR) {
//		int size = S->ext_size;
//		for (int layer = 0; layer < nlayer; ++layer) {
//			S->EXT_RHS[i1 + layer * size] /= S->EXT_D[i1 + layer * size];
//		}
//		for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
//			for (int layer = 0; layer < nlayer; ++layer) {
//				S->EXT_RHS[nrn_seg + layer * size] -= S->EXT_B[nrn_seg + layer * size] * S->EXT_RHS[nrn_seg - 1 + layer * size];
//				S->EXT_RHS[nrn_seg + layer * size] /= S->EXT_D[nrn_seg + layer * size];
//			}
//		}
//	}
}

__device__
void triang(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void triang(NrnThread* _nt)
	 */
	// intracellular
	double ppp;
	for (int nrn_seg = i3 - 1; nrn_seg >= i1 + 1; --nrn_seg) {
		ppp = S->NODE_A[nrn_seg] / S->NODE_D[nrn_seg];
		S->NODE_D[nrn_seg - 1] -= ppp * S->NODE_B[nrn_seg];
		S->NODE_RHS[nrn_seg - 1] -= ppp * S->NODE_RHS[nrn_seg];
	}
	// extracellular
//	if (EXTRACELLULAR) {
//		int size = S->ext_size;
//		for (int nrn_seg = i3 - 1; nrn_seg >= i1 + 1; --nrn_seg) {
//			for (int layer = 0; layer < nlayer; ++layer) {
//				ppp = S->EXT_A[nrn_seg + layer * size] / S->EXT_D[nrn_seg + layer * size];
//				S->EXT_D[nrn_seg - 1 + layer * size] -= ppp * S->EXT_B[nrn_seg + layer * size];
//				S->EXT_RHS[nrn_seg - 1 + layer * size] -= ppp * S->EXT_RHS[nrn_seg + layer * size];
//				nrn_seg--;
//			}
//		}
//	}
}

__device__
void nrn_solve(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void nrn_solve(NrnThread* _nt)
	 */
#ifdef LOG
	printf("SOLVE EXT begin NODED 0 "); for (int i=i1; i < i3; ++i) printf("%g\t", S->EXT_D[i]); printf("\n");
	printf("SOLVE EXT begin NODED 0 "); for (int i=i1; i < i3; ++i) printf("%g\t", S->EXT_D[i + S->ext_size]); printf("\n");
#endif
	// TODO PROOVED
	// nrn_solve EXT D 0 5e+07	2e-09	2e-09	1e-09	5e+07
	// nrn_solve EXT D 1 5e+07	5e-10	5e-10	1e-09	5e+07
	triang(S, P, N, nrn, i1, i3);
	bksub(S, P, N, nrn, i1, i3);
}

__device__
void setup_tree_matrix(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void setup_tree_matrix(NrnThread* _nt)
	 */
	nrn_rhs(S, P, N, nrn, i1, i3);
	nrn_lhs(S, P, N, nrn, i1, i3);
}

__device__
void update(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void update(NrnThread* _nt)
	 */
#ifdef LOG
	printf("UPDATE EXT begin NODED 0 "); for (int i=i1; i < i3; ++i) printf("%g\t", S->EXT_D[i]); printf("\n");
	printf("UPDATE EXT begin NODED 1 "); for (int i=i1; i < i3; ++i) printf("%g\t", S->EXT_D[i + S->ext_size]); printf("\n");
	// final voltage updating
#endif
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->Vm[nrn_seg] += S->NODE_RHS[nrn_seg];
	}
	// extracellular
	if (EXTRACELLULAR) {
		nrn_update_2d(S, P, N, i1, i3);
	}
}

__device__
void nrn_deliver_events(States* S, const Parameters* P, Neurons* N, int nrn) {
	/**
	 * void nrn_deliver_events(NrnThread* nt)
	 */
	// get the central segment (for detecting spikes): i1 + (2 or 1)
	int seg_update = P->nrn_start_seg[nrn] + ((P->models[nrn] == MUSCLE)? 2 : 1);
	// check if neuron has spike with special flag for avoidance multi-spike detecting
	if (!N->spike_on[nrn] && S->Vm[seg_update] > V_th && N->ref_time_timer[nrn] == 0) {
		N->spike_on[nrn] = true;
		N->has_spike[nrn] = true;
		N->ref_time_timer[nrn] = N->ref_time[nrn];
	} else if (S->Vm[seg_update] < V_th) {
		N->spike_on[nrn] = false;
	}
}

__device__
void nrn_fixed_step_lastpart(States* S, const Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void *nrn_fixed_step_lastpart(NrnThread *nth)
	 */
	// update neurons' synapses state
	recalc_synaptic(S, P, N, nrn);
	//  update neurons' segments state
	if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
		for(int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			recalc_inter_channels(S, P, N, nrn_seg, S->Vm[nrn_seg]);
		}
	} else if (P->models[nrn] == MOTO) {
		for(int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			recalc_moto_channels(S, P, N, nrn_seg, S->Vm[nrn_seg]);
		}
	} else if (P->models[nrn] == MUSCLE) {
		for(int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			recalc_muslce_channels(S, P, N, nrn_seg, S->Vm[nrn_seg]);
		}
	} else {
		assert(false);
	}
	//  spike detection for (in synapse kernel)
	nrn_deliver_events(S, P, N, nrn);
}

__device__
void nrn_area_ri(States* S, const Parameters* P, Neurons* N) {
	/**
	 * void nrn_area_ri(Section *sec) [790] treeset.c
	 * area for right circular cylinders. Ri as right half of parent + left half of this
	 */
	printf("GPU: nrn_area_ri\n");
	double dx, rleft, rright;
	int i1, i3, nrn_seg, segments;
	//
	for (int nrn = 0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		segments = (i3 - i1 - 2);
		dx = P->length[nrn] / segments; // divide by the last index of node (or segments count)
		rright = 0;
		// todo sec->pnode needs +1 index
		for (nrn_seg = i1 + 1; nrn_seg < i1 + segments + 1; ++nrn_seg) {
			// area for right circular cylinders. Ri as right half of parent + left half of this
			S->NODE_AREA[nrn_seg] = PI * dx * P->diam[nrn];
			rleft = 1.e-2 * P->Ra[nrn] * (dx / 2.0) / (PI * pow(P->diam[nrn], 2) / 4.0);   // left half segment Megohms
			S->NODE_RINV[nrn_seg] = 1.0 / (rleft + rright); // uS
			rright = rleft;
		}
		//the first and last segments has zero length. Area is 1e2 in dimensionless units
		S->NODE_AREA[i1] = 100.0;
		nrn_seg = i1 + segments + 1; // the last segment
		S->NODE_AREA[nrn_seg] = 100.0;
		S->NODE_RINV[nrn_seg] = 1.0 / rright;
	}
}

__device__
void ext_con_coef(States* S, const Parameters* P, Neurons* N) {
	/**
	 * void ext_con_coef(void)
	 * setup a and b
	 */
	double dx;
	int layer, i1, i3, segments, size = S->ext_size;
	// todo: extracellular only for those neurons who need
	for (int nrn = 0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		layer = 0;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		segments = (i3 - i1 - 2);
		// temporarily store half segment resistances in rhs
		for (int nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
			dx = P->length[nrn] / segments;
			S->EXT_RHS[nrn_seg + layer * size] = 1e-4 * xraxial * dx / 2;  // Megohms
		}
		// last segment has 0 length
		S->EXT_RHS[i3 - 1 + layer * size] = 0; // todo i3 -1 or just i3
		// NEURON RHS = [5e+07 5e+07 5e+07 0 ]
		// GRAS RHS = [0 5e+07 5e+07 5e+07 0 ]
#ifdef LOG
		printf("EXT RHS: "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_RHS[ii]); printf("\n");
#endif
		// node half resistances in general get added to the node and to the node's "child node in the same section".
		// child nodes in different sections don't involve parent node's resistance
		S->EXT_B[i1 + 1 + layer * size] = S->EXT_RHS[i1 + 1 + layer * size];
		for (int nrn_seg = i1 + 1 + 1; nrn_seg < i3; ++nrn_seg) {
			S->EXT_B[nrn_seg + layer * size] = S->EXT_RHS[nrn_seg + layer * size] + S->EXT_RHS[nrn_seg - 1 + layer * size]; // Megohms
		}
		// NEURON B = [5e+07 1e+08 1e+08 5e+07 ]
		// GRAS B = [0 5e+07 1e+08 1e+08 5e+07 ]
#ifdef LOG

		printf("EXT B: "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_B[ii]); printf("\n");
#endif
		// first the effect of node on parent equation. Note That last nodes have area = 1.e2 in
		// dimensionless units so that last nodes have units of microsiemens's
		double area = S->NODE_AREA[i1]; // parentnode index of sec is 0
		double rall_branch = 1.0;  // sec->prop->dparam[4].val
		S->EXT_A[i1 + 1 + layer * size] = -1.e2 * rall_branch / (S->EXT_B[i1 + 1 + layer * size] * area);
		for (int nrn_seg = i1 + 1 + 1; nrn_seg < i3; ++nrn_seg) {
			area = S->NODE_AREA[nrn_seg - 1];
			S->EXT_A[nrn_seg + layer * size] = -1.e2 / (S->EXT_B[nrn_seg + layer * size] * area);
		}
		// NEURON A = [-2e-08 -7.95775e-12 -7.95775e-12 -1.59155e-11 ]
		// GRAS A = [0 -2e-08 -7.95775e-12 -7.95775e-12 -1.59155e-11 ]
#ifdef LOG

		printf("EXT A: "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_A[ii]); printf("\n");
#endif
		// now the effect of parent on node equation
		// todo sec->pnode needs +1 index
		for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
			S->EXT_B[nrn_seg + layer * size] = -1.e2 / (S->EXT_B[nrn_seg + layer * size] * S->NODE_AREA[nrn_seg]);
		}
		// NEURON B = [-1.59155e-11 -7.95775e-12 -7.95775e-12 -2e-08 ]
		// GRAS B = [0 -1.59155e-11 -7.95775e-12 -7.95775e-12 -2e-08 ]
#ifdef LOG

		printf("EXT B END: "); for(int ii = i1; ii < i3; ++ii) printf("%g\t", S->EXT_B[ii]); printf("\n");
#endif
		// the same for other layers
		for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			S->EXT_A[nrn_seg + 1 * size] = S->EXT_A[nrn_seg + 0 * size];
			S->EXT_B[nrn_seg + 1 * size] = S->EXT_B[nrn_seg + 0 * size];
			S->EXT_RHS[nrn_seg + 1 * size] = S->EXT_RHS[nrn_seg + 0 * size];
		}
	}
}

__device__
void connection_coef(States* S, const Parameters* P, Neurons* N) {
	/**
	 * void connection_coef(void) treeset.c
	 */
	printf("GPU: connection_coef\n");
	nrn_area_ri(S, P, N);
	int i1, i3, nrn_seg, segments;
	//
	for (int nrn = 0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		segments = (i3 - i1 - 2);
		// first the effect of node on parent equation. Note that last nodes have area = 1.e2 in dimensionless
		// units so that last nodes have units of microsiemens
		// todo sec->pnode needs +1 index
		nrn_seg = i1 + 1;
		// sec->prop->dparam[4].val = 1, what is dparam[4].val
		S->NODE_A[nrn_seg] = -1.e2 * 1.0 * S->NODE_RINV[nrn_seg] / S->NODE_AREA[nrn_seg - 1];
		// todo sec->pnode needs +1 index
		for (nrn_seg = i1 + 1 + 1; nrn_seg < i1 + segments + 1 + 1; ++nrn_seg) {
			S->NODE_A[nrn_seg] = -1.e2 * S->NODE_RINV[nrn_seg] / S->NODE_AREA[nrn_seg - 1];
		}
		// now the effect of parent on node equation
		// todo sec->pnode needs +1 index
		for (nrn_seg = i1 + 1; nrn_seg < i1 + segments + 1 + 1; ++nrn_seg) {
			S->NODE_B[nrn_seg] = -1.e2 * S->NODE_RINV[nrn_seg] / S->NODE_AREA[nrn_seg];
		}
	}
	// for extracellular
	if (EXTRACELLULAR) {
		ext_con_coef(S, P, N);
	}

	/**
	 * note: from LHS, this functions just recalc each time the constant NODED (!)
	 * void nrn_lhs(NrnThread *_nt)
	 * NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	 */
	// nt->cj = 2/dt if (secondorder) else 1/dt
	// note, the first is CAP
	// function nrn_cap_jacob(_nt, _nt->tml->ml);
	double cj = 1.0 / dt;
	double cfac = 0.001 * cj;
	for (int nrn = 0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		// nrn_cap_jacob
		for (nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
			S->const_NODE_D[nrn_seg] += cfac * P->Cm[nrn];
		}
	}
}

__global__
void initialization_kernel(curandState *state, States* S, const Parameters* P, Neurons* N, double v_init) {
	/**
	 *
	 */
	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		int i1, i3;
		printf("GPU: initialization_kernel\n");
		//
		connection_coef(S, P, N);
		// for different models -- different init function
		for (int nrn = 0; nrn < N->size; ++nrn) {
			// init random
			curand_init(7 + nrn, nrn, 0, &state[nrn]);

			// do not init neuron state for generator
			if (P->models[nrn] == GENERATOR)
				continue;
			i1 = P->nrn_start_seg[nrn];
			i3 = P->nrn_start_seg[nrn + 1];
			// for each segment init the neuron model
			for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
				S->Vm[nrn_seg] = v_init;
				if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
					nrn_inter_initial(S, P, N, nrn_seg, v_init);
				} else if (P->models[nrn] == MOTO) {
					nrn_moto_initial(S, P, N, nrn_seg, v_init);
				} else if (P->models[nrn] == MUSCLE) {
					nrn_muslce_initial(S, P, N, nrn_seg, v_init);
				} else {
					assert(false);
				}
			}
			// init RHS/LHS
			setup_tree_matrix(S, P, N, nrn, i1, i3);
			// init tau synapses
			syn_initial(S, P, N, nrn);
		}
	}
}

void conn_generator(Group &generator, Group &post_neurons, double delay, double weight, int indegree=50) {
	/**
	 *
	 */
	uniform_int_distribution<int> nsyn_distr(indegree, indegree + 5);
	normal_distribution<double> delay_distr(delay, delay / 5);
	normal_distribution<double> weight_distr(weight, weight / 6);

//	int nsyn = nsyn_distr(rand_gen);
	printf("Connect generator %s [%d..%d] to %s [%d..%d] (1:%d). Synapses %d, D=%.1f, W=%.2f\n",
	       generator.group_name.c_str(), generator.id_start, generator.id_end,
	       post_neurons.group_name.c_str(), post_neurons.id_start, post_neurons.id_end,
	       post_neurons.group_size, generator.group_size * post_neurons.group_size, delay, weight);
	//
	int nsyn = nsyn_distr(rand_gen);
	int gen_id = generator.id_start;
	if (generator.group_size > 1) {
		printf("Generator cannot include more than 1 neuron!\n");
		exit(0);
	}
	for (int post = post_neurons.id_start; post <= post_neurons.id_end; ++post) {
		for (int i = 0; i < nsyn; ++i) {
			vector_syn_pre_nrn.push_back(gen_id);
			vector_syn_post_nrn.push_back(post);
			vector_syn_weight.push_back(weight_distr(rand_gen));
			vector_syn_delay.push_back(ms_to_step(delay_distr(rand_gen)));
			vector_syn_delay_timer.push_back(-1);
		}
	}
}

void connect_fixed_indegree(Group &pre_neurons, Group &post_neurons, double delay, double weight, int indegree=50, short high_distr=0) {
	/**
	 *
	 */
	// STR
	if (str_flag && weight < 0)
		weight = 0;

	if (post_neurons.model == INTER) {
		printf("POST INTER ");
		weight /= 11.0;
	}

	uniform_int_distribution<int> nsyn_distr(indegree - 15, indegree);
	uniform_int_distribution<int> pre_nrns_ids(pre_neurons.id_start, pre_neurons.id_end);
	double d_spread, w_spread;
	double d_left, d_right, w_left, w_right = 0;
	if (high_distr == 0) {
		d_spread = 0; //delay / 6;
		w_spread = 0; //weight / 6;
	} else if (high_distr == 1) {
		d_spread = delay / 5;
		w_spread = weight / 5.5;
	} else if (high_distr == 2) {
		d_spread = delay / 3.5;
		w_spread = weight / 2.5;
	} else if (high_distr == 3) {
		d_spread = delay / 1.2;
		w_spread = weight / 1.1;

		d_left = delay - d_spread;
		d_right = delay + d_spread;

		w_left = weight - w_spread;
		w_right = weight + w_spread + w_spread / 2;
	} else if (high_distr == 4) {
		d_spread = delay / 3;
		w_spread = weight / 3;

		d_left = delay - d_spread;
		d_right = delay + d_spread;

		w_left = weight - w_spread;
		w_right = weight + w_spread + w_spread / 2;
	} else if (high_distr == 5) {
		d_spread = delay / 1.1;
		w_spread = weight / 1.1;

		d_left = delay - d_spread;
		d_right = delay + d_spread + delay * 1.5;

		w_left = weight - w_spread;
		w_right = weight + w_spread + w_spread;
	}else {
		logic_error("distr only 0 1 2");
	}
	normal_distribution<double> delay_distr(delay, d_spread);
	normal_distribution<double> weight_distr(weight, w_spread);
	uniform_real_distribution<double> delay_distr_U(d_left, d_right);
	uniform_real_distribution<double> weight_distr_U(w_left, w_right);

	auto nsyn = nsyn_distr(rand_gen);

	printf("Connect indegree %s [%d..%d] to %s [%d..%d] (1:%d). Synapses %d, D=%.1f, W=%.2f\n",
	       pre_neurons.group_name.c_str(), pre_neurons.id_start, pre_neurons.id_end,
	       post_neurons.group_name.c_str(), post_neurons.id_start, post_neurons.id_end,
	       indegree, post_neurons.group_size * indegree, delay, weight);
	//
	int prerand = 0;
	double tmp_w = 0;
	double tmp_d = 0;
	for (int post = post_neurons.id_start; post <= post_neurons.id_end; ++post) {
		for (int i = 0; i < nsyn; ++i) {
			prerand = pre_nrns_ids(rand_gen);
			vector_syn_pre_nrn.push_back(prerand);
			vector_syn_post_nrn.push_back(post);
			if (post_neurons.model == AFFERENTS) {
				vector_syn_weight.push_back(weight);
				vector_syn_delay.push_back(ms_to_step(delay));
			} else {
				if (high_distr == 3 || high_distr == 4 || high_distr == 5) {
					tmp_w = weight_distr_U(rand_gen);
					tmp_d = delay_distr_U(rand_gen);
				} else {
					tmp_w = weight_distr(rand_gen);
					if (tmp_w <= 0) {
						tmp_w = weight;
					}
					tmp_d = delay_distr(rand_gen);
					if (tmp_d <= 0.01) {
						tmp_d = delay;
					}
				}
				vector_syn_weight.push_back(tmp_w);
				vector_syn_delay.push_back(ms_to_step(tmp_d));
			}
			vector_syn_delay_timer.push_back(-1);
		}
	}
}


void connect_fixed_outdegree_MUSCLE(Group &pre_neurons, Group &post_neurons, double delay, double weight, int indegree=50, short high_distr=0) {
	/**
	 *
	 */
	uniform_int_distribution<int> nsyn_distr(indegree - 15, indegree);

//	double d_spread, w_spread;
//	if (high_distr == 0) {
//		d_spread = 0;//delay / 6;
//		w_spread = 0;//weight / 6;
//	} else if (high_distr == 1) {
//		d_spread = delay / 5;
//		w_spread = weight / 5.5;
//	} else if (high_distr == 2) {
//		d_spread = delay / 4;
//		w_spread = weight / 4;
//	}else {
//		logic_error("distr only 0 1 2");
//	}
//	normal_distribution<double> delay_distr(delay, d_spread);
//	normal_distribution<double> weight_distr(weight, w_spread);



	auto nsyn = nsyn_distr(rand_gen);

	printf("Connect indegree %s [%d..%d] to %s [%d..%d] (1:%d). Synapses %d, D=%.1f, W=%.2f\n",
	       pre_neurons.group_name.c_str(), pre_neurons.id_start, pre_neurons.id_end,
	       post_neurons.group_name.c_str(), post_neurons.id_start, post_neurons.id_end,
	       indegree, post_neurons.group_size * indegree, delay, weight);
	//
	int shift, post_rand = 0;
	double tmp_w, tmp_d = 0;
	double d_left, d_right, w_left, w_right, d_spread, w_spread = 0;

	if (high_distr == 5) {
		d_spread = delay / 1.1;
		w_spread = weight / 1.1;

		d_left = delay - d_spread;
		d_right = delay + d_spread + delay * 1.5;

		w_left = weight - w_spread;
		w_right = weight + w_spread + w_spread;
	} else {
		d_left = delay;
		d_right = delay + 3;

		w_left = weight - weight / 1.5;
		w_right = weight;
	}

	uniform_real_distribution<double> delay_distr_U(d_left, d_right);
	uniform_real_distribution<double> weight_distr_U(w_left, w_right);

	int m_start = post_neurons.id_start;
	for (int pre = pre_neurons.id_start; pre <= pre_neurons.id_end; ++pre) {
		uniform_int_distribution<int> post_nrns_ids(m_start + 50 * shift, m_start + 50 * (shift + 1));
		for (int i = 0; i < nsyn; ++i) {
			post_rand = post_nrns_ids(rand_gen);
			vector_syn_pre_nrn.push_back(pre);
			vector_syn_post_nrn.push_back(post_rand);
//			tmp_w = weight_distr(rand_gen);
			tmp_w = weight_distr_U(rand_gen);
			if (tmp_w <= 0)
				tmp_w = weight;
//			tmp_d = delay_distr(rand_gen);
			tmp_d = delay_distr_U(rand_gen);
			if (tmp_d <= 0.01)
				tmp_d = delay;
			vector_syn_weight.push_back(tmp_w);
			vector_syn_delay.push_back(ms_to_step(tmp_d));
			vector_syn_delay_timer.push_back(-1);
		}
	}
}

void connect_fixed_outdegree(Group &pre_neurons, Group &post_neurons, double delay, double weight, int outdegree=50, short high_distr=0) {
	/**
	 *
	 */
	// STR
	if (weight < 0)
		weight /= 1000;

	if (post_neurons.model == INTER) {
		printf("POST INTER ");
		weight /= 11.0;
	}

	uniform_int_distribution<int> nsyn_distr(outdegree - 15, outdegree);
	uniform_int_distribution<int> post_nrns_ids(post_neurons.id_start, post_neurons.id_end);
	double d_spread, w_spread;
	if (high_distr == 0) {
		d_spread = delay / 6;
		w_spread = weight / 6;
	} else if (high_distr == 1) {
		d_spread = delay / 5;
		w_spread = weight / 5.5;
	} else if (high_distr == 2) {
		d_spread = delay / 3.5;
		w_spread = weight / 3.5;
	} else if (high_distr == 3) {
		d_spread = delay / 4;
		w_spread = weight / 4;
	} else {
		logic_error("distr only 0 1 2");
	}
	normal_distribution<double> delay_distr(delay, d_spread);
	normal_distribution<double> weight_distr(weight, w_spread);
	auto nsyn = nsyn_distr(rand_gen);

	printf("Connect OUTdegree %s [%d..%d] to %s [%d..%d] (1:%d). Synapses %d, D=%.1f, W=%.2f\n",
	       pre_neurons.group_name.c_str(), pre_neurons.id_start, pre_neurons.id_end,
	       post_neurons.group_name.c_str(), post_neurons.id_start, post_neurons.id_end,
	       outdegree, post_neurons.group_size * outdegree, delay, weight);
	//
	int postrand = 0;
	for (int pre = pre_neurons.id_start; pre <= pre_neurons.id_end; ++pre) {
		for (int i = 0; i < nsyn; ++i) {
			postrand = post_nrns_ids(rand_gen);
			vector_syn_pre_nrn.push_back(pre);
			vector_syn_post_nrn.push_back(postrand);
			if (post_neurons.model == AFFERENTS) {
				vector_syn_weight.push_back(weight);
				vector_syn_delay.push_back(ms_to_step(delay));
			} else {
				vector_syn_weight.push_back(weight_distr(rand_gen));
				vector_syn_delay.push_back(ms_to_step(delay_distr(rand_gen)));
			}
			vector_syn_delay_timer.push_back(-1);
		}
	}
}

void connectinsidenucleus(Group &nucleus) {
	connect_fixed_indegree(nucleus, nucleus, 0.5, 0.25, 50, 3);
}

void file_writing(int test_index
                  , GroupMetadata &metadata, const string &folder) {
	/**
	 *
	 */
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
	for (double const &value: metadata.spike_vector) {
		file << value << " ";
	}
	file.close();

	cout << "Saved to: " << folder + file_name << endl;
}

void save(vector<Group> groups) {
	for (Group &group : groups)
		saving_groups.emplace_back(GroupMetadata(group, SIM_TIME_IN_STEPS));
}

void copy_data_to(GroupMetadata& metadata,
                  const double* Vm,
                  double* tmp,
                  const double* g_exc,
                  const double* g_inh_A,
                  const double* g_inh_B,
                  const bool* has_spike,
                  const unsigned int sim_iter) {
	double nrn_mean_volt = 0;
	double nrn_mean_g_exc = 0;
	double nrn_mean_g_inh = 0;

	int center;
	unsigned int id_start = metadata.group.id_start;
	unsigned int id_end = metadata.group.id_end;

	short shift = (vector_models[id_start] == MUSCLE) ? 2 : 1;

	if (metadata.group.group_name == "muscle_E" || metadata.group.group_name == "muscle_F") {
//		#pragma omp parallel default(none) shared(vector_nrn_start_seg, Vm, tmp, nrn_mean_volt, id_start, id_end, shift)
//		#pragma omp for reduction(+:nrn_mean_volt) private(center) schedule(auto)
		for (auto nrn = id_start; nrn <= id_end; ++nrn) {
			center = vector_nrn_start_seg[nrn] + shift;
			nrn_mean_volt += (Vm[center] - tmp[nrn]);
			tmp[nrn] = Vm[center];
			nrn_mean_g_exc += g_exc[nrn];
			nrn_mean_g_inh += (g_inh_B[nrn] - g_inh_A[nrn]);
			if (has_spike[nrn]) {
				metadata.spike_vector.push_back(step_to_ms(sim_iter));
			}
		}
		metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size / dt * (4 * PI * 10000);
	} else {
		for (unsigned int nrn = id_start; nrn <= id_end; ++nrn) {
			center = vector_nrn_start_seg[nrn] + shift;
			nrn_mean_volt += Vm[center];
			nrn_mean_g_exc += g_exc[nrn];
			nrn_mean_g_inh += (g_inh_B[nrn] - g_inh_A[nrn]);
			if (has_spike[nrn]) {
				metadata.spike_vector.push_back(step_to_ms(sim_iter));
			}
		}
		metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size;
	}
	metadata.g_exc[sim_iter] = nrn_mean_g_exc / metadata.group.group_size;
	metadata.g_inh[sim_iter] = nrn_mean_g_inh / metadata.group.group_size;
}


void save_result(int test_index) {
	string current_path = getcwd(nullptr, 0);
	for (GroupMetadata &metadata : saving_groups)
		file_writing(test_index, metadata, current_path);
	printf("[Test #%d] Saved results to: %s \n", test_index, current_path.c_str());
}

template<typename type>
type* arr_init(int size = NRNS_AND_SEGS) {
	// important: NRNS_AND_SEGS initialized at network building
	return new type[size]();
}

void createmotif(Group &OM0, Group &OM1, Group &OM2, Group &OM3) {
	/**
	 * Connects motif module
	 * see https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
	 */
	connect_fixed_indegree(OM0, OM1, 3, 0.9, 50, 5);
	connect_fixed_indegree(OM1, OM2, 3, 0.6, 50, 5); // 0.85
	connect_fixed_indegree(OM2, OM1, 4, 0.6, 50, 5);
	connect_fixed_indegree(OM1, OM3, 2, 0.0003); // 2.5
	connect_fixed_indegree(OM2, OM3, 2, 0.0005); // 2.5
	connect_fixed_indegree(OM3, OM2, 2, -3);
	connect_fixed_indegree(OM3, OM1, 2, -3);
}

void createmotif_flexor(Group &OM0, Group &OM1, Group &OM2, Group &OM3) {
	connect_fixed_indegree(OM0, OM1, 3, 0.9, 50, 5);
	connect_fixed_indegree(OM1, OM2, 3, 0.8, 50, 5); // 0.85
	connect_fixed_indegree(OM2, OM1, 4, 0.6, 50, 5);
	connect_fixed_indegree(OM1, OM3, 2, 0.0002); // 2.5
	connect_fixed_indegree(OM2, OM3, 2, 0.0004); // 4
	connect_fixed_indegree(OM3, OM2, 2, -1); // -1 - noise, -5 - void
	connect_fixed_indegree(OM3, OM1, 2, -3);
}

__global__
void neuron_kernel(curandState *state, States *S, const Parameters *P, Neurons *N, Generators *G, int t) {
	/**
	 *
	 */
	int i1, i3;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//
	for (int nrn = tid; nrn < N->size; nrn += blockDim.x * gridDim.x) {
		// reset the spike state
		N->has_spike[nrn] = false;
		//
		if (P->models[nrn] != GENERATOR) {
			// calc the borders of the neuron by theirs segments
			i1 = P->nrn_start_seg[nrn];
			i3 = P->nrn_start_seg[nrn + 1];
			// generate pseudo-random noise
			// re-calc currents and states based on synaptic activity
			setup_tree_matrix(S, P, N, nrn, i1, i3);
			// solve equations
			nrn_solve(S, P, N, nrn, i1, i3);
			// change voltage of the neurons based on solved equations
			update(S, P, N, nrn, i1, i3);
			// recalc conductance, update channels and deliver network events
			nrn_fixed_step_lastpart(S, P, N, nrn, i1, i3);
			if (N->ref_time_timer[nrn] > 0)
				N->ref_time_timer[nrn]--;
		}
	}
	// update generators
	if (tid == 0) {
		for (int generator = 0; generator < G->size; ++generator) {
			if (t == G->spike_each_step[generator] && t < G->time_end[generator]) {
				G->spike_each_step[generator] += G->freq_in_steps[generator];
				N->has_spike[G->nrn_id[generator]] = true;
			}
		}
		// afferent
//		float part;
//		if ( ((25 / dt <= t) && (t < 50 / dt)) || ((150 / dt <= t) && (t < 175 / dt)) ) {
//			part = (2678 - 2559) * 0.4;
//			for (int n = 2559; n <= 2678 - part; n += 2)
//				N->has_spike[n] = false;
//		} else if ( ((50 / dt <= t) && (t < 75 / dt)) || ((125 / dt <= t) && (t < 150 / dt)) ) {
//			part = (2678 - 2559) * 0.6;
//			for (int n = 2559; n <= 2678 - part; n += 3)
//				N->has_spike[n] = false;
//		}
	}
}

__global__
void synapse_kernel(Neurons *N, Synapses* synapses) {
	/**
	 * void deliver_net_events(NrnThread* nt)
	 */
	int pre_nrn, post_id;
	double weight;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < synapses->size; index += blockDim.x * gridDim.x) {
		pre_nrn = synapses->syn_pre_nrn[index];
		// synapse update
		if (synapses->syn_delay_timer[index] > 0) {
			synapses->syn_delay_timer[index]--;
		} else {
			// if timer is over -> synapse change the conductance of the post neuron
			if (synapses->syn_delay_timer[index] == 0) {
				post_id = synapses->syn_post_nrn[index];
				weight = synapses->syn_weight[index];
				if (weight >= 0) {
					if (N->ref_time_timer[post_id] == 0)
						atomicAdd(&N->g_exc[post_id], weight);
				} else {
					atomicAdd(&N->g_inh_A[post_id], -weight * N->factor[post_id]);
					atomicAdd(&N->g_inh_B[post_id], -weight * N->factor[post_id]);
				}
				synapses->syn_delay_timer[index] = -1;
			} else {
				// if pre nrn has spike and synapse is ready to send siagnal
				if (N->has_spike[pre_nrn] && synapses->syn_delay_timer[index] == -1) {
					synapses->syn_delay_timer[index] = synapses->syn_delay[index];
				}
			}
		}
	}
}
void simulate(void (*network)()) {
    /**
     *
     */
    // init structs (CPU)
    States *S = (States *)malloc(sizeof(States));
    Parameters *P = (Parameters *)malloc(sizeof(Parameters));
    Neurons *N = (Neurons *)malloc(sizeof(Neurons));
    Synapses *synapses = (Synapses *)malloc(sizeof(Synapses));
    Generators *G = (Generators *)malloc(sizeof(Generators));

    // create neurons and their connectomes
    network();
    // note: important
    vector_nrn_start_seg.push_back(NRNS_AND_SEGS);   //добавляет ноль?

    // allocate generators into the GPU
    unsigned int gens_number = vec_spike_each_step.size();
    G->nrn_id = init_gpu_arr(vec_nrn_id);
    G->time_end = init_gpu_arr(vec_time_end);
    G->freq_in_steps = init_gpu_arr(vec_freq_in_steps);
    G->spike_each_step = init_gpu_arr(vec_spike_each_step);
    G->size = gens_number;

    // allocate static parameters into the GPU
    P->nrn_start_seg = init_gpu_arr(vector_nrn_start_seg);
    P->models = init_gpu_arr(vector_models);
    P->Cm = init_gpu_arr(vector_Cm);
    P->gnabar = init_gpu_arr(vector_gnabar);
    P->gkbar = init_gpu_arr(vector_gkbar);
    P->gl = init_gpu_arr(vector_gl);
    P->Ra = init_gpu_arr(vector_Ra);
    P->diam = init_gpu_arr(vector_diam);
    P->length = init_gpu_arr(vector_length);
    P->ena = init_gpu_arr(vector_ena);
    P->ek = init_gpu_arr(vector_ek);
    P->el = init_gpu_arr(vector_el);
    P->gkrect = init_gpu_arr(vector_gkrect);
    P->gcaN = init_gpu_arr(vector_gcaN);
    P->gcaL = init_gpu_arr(vector_gcaL);
    P->gcak = init_gpu_arr(vector_gcak);
    P->E_ex = init_gpu_arr(vector_E_ex);
    P->E_inh = init_gpu_arr(vector_E_inh);
    P->tau_exc = init_gpu_arr(vector_tau_exc);
    P->tau_inh1 = init_gpu_arr(vector_tau_inh1);
    P->tau_inh2 = init_gpu_arr(vector_tau_inh2);
    P->size = NRNS_NUMBER;

    // dynamic states of neuron (CPU arrays) and allocate them into the GPU
//	double *Vm; HANDLE_ERROR(cudaMallocHost((void**)&Vm, NRNS_AND_SEGS));
    auto *Vm = arr_init<double>(); S->Vm = init_gpu_arr(Vm);
    auto *n = arr_init<double>(); S->n = init_gpu_arr(n);
    auto *m = arr_init<double>(); S->m = init_gpu_arr(m);
    auto *h = arr_init<double>(); S->h = init_gpu_arr(h);
    auto *l = arr_init<double>(); S->l = init_gpu_arr(l);
    auto *s = arr_init<double>(); S->s = init_gpu_arr(s);
    auto *p = arr_init<double>(); S->p = init_gpu_arr(p);
    auto *hc = arr_init<double>(); S->hc = init_gpu_arr(hc);
    auto *mc = arr_init<double>(); S->mc = init_gpu_arr(mc);
    auto *cai = arr_init<double>(); S->cai = init_gpu_arr(cai);
    auto *I_Ca = arr_init<double>(); S->I_Ca = init_gpu_arr(I_Ca);
    auto *NODE_A = arr_init<double>(); S->NODE_A = init_gpu_arr(NODE_A);
    auto *NODE_B = arr_init<double>(); S->NODE_B = init_gpu_arr(NODE_B);
    auto *NODE_D = arr_init<double>(); S->NODE_D = init_gpu_arr(NODE_D);
    auto *const_NODE_D = arr_init<double>(); S->const_NODE_D = init_gpu_arr(const_NODE_D);
    auto *NODE_RHS = arr_init<double>(); S->NODE_RHS = init_gpu_arr(NODE_RHS);
    auto *NODE_RINV = arr_init<double>(); S->NODE_RINV = init_gpu_arr(NODE_RINV);
    auto *NODE_AREA = arr_init<double>(); S->NODE_AREA = init_gpu_arr(NODE_AREA);

//	int ext_size = NRNS_AND_SEGS * 2;
//	auto *EXT_A = arr_init<double>(ext_size); S->EXT_A = init_gpu_arr(EXT_A, ext_size);
//	auto *EXT_B = arr_init<double>(ext_size); S->EXT_B = init_gpu_arr(EXT_B, ext_size);
//	auto *EXT_D = arr_init<double>(ext_size); S->EXT_D = init_gpu_arr(EXT_D, ext_size);
//	auto *EXT_V = arr_init<double>(ext_size); S->EXT_V = init_gpu_arr(EXT_V, ext_size);
//	auto *EXT_RHS = arr_init<double>(ext_size); S->EXT_RHS = init_gpu_arr(EXT_RHS, ext_size);
    S->size = NRNS_AND_SEGS;
//	S->ext_size = ext_size;

    // special neuron's state (CPU) and allocate them into the GPU
    auto *tmp = arr_init<double>(NRNS_NUMBER);
    for (int i = 0; i < NRNS_NUMBER; ++i)
        tmp[i] = 0.0;

    auto *has_spike = arr_init<bool>(NRNS_NUMBER); N->has_spike = init_gpu_arr(has_spike, NRNS_NUMBER);
    auto *g_exc = arr_init<double>(NRNS_NUMBER); N->g_exc = init_gpu_arr(g_exc, NRNS_NUMBER);
    auto *g_inh_A = arr_init<double>(NRNS_NUMBER); N->g_inh_A = init_gpu_arr(g_inh_A, NRNS_NUMBER);
    auto *g_inh_B = arr_init<double>(NRNS_NUMBER); N->g_inh_B = init_gpu_arr(g_inh_B, NRNS_NUMBER);
    auto *spike_on = arr_init<bool>(NRNS_NUMBER); N->spike_on = init_gpu_arr(spike_on, NRNS_NUMBER);
    auto *factor = arr_init<double>(NRNS_NUMBER); N->factor = init_gpu_arr(factor, NRNS_NUMBER);
    auto *ref_time_timer = arr_init<unsigned int>(NRNS_NUMBER); N->ref_time_timer = init_gpu_arr(ref_time_timer, NRNS_NUMBER);
    auto *ref_time = arr_init<unsigned int>(NRNS_NUMBER);
    for (int i = 0; i < NRNS_NUMBER; ++i)
        ref_time[i] = ms_to_step(2);
    N->ref_time = init_gpu_arr(ref_time, NRNS_NUMBER);
    N->size = NRNS_NUMBER;

    // synaptic parameters
    unsigned int synapses_number = vector_syn_delay.size();
    synapses->syn_pre_nrn = init_gpu_arr(vector_syn_pre_nrn);
    synapses->syn_post_nrn = init_gpu_arr(vector_syn_post_nrn);
    synapses->syn_weight = init_gpu_arr(vector_syn_weight);
    synapses->syn_delay = init_gpu_arr(vector_syn_delay);
    synapses->syn_delay_timer = init_gpu_arr(vector_syn_delay_timer);
    synapses->size = synapses_number;

    // allocate structs to the device
    auto *dev_S = init_gpu_arr(S, 1);
    auto *dev_P = init_gpu_arr(P, 1);
    auto *dev_N = init_gpu_arr(N, 1);
    auto *dev_G = init_gpu_arr(G, 1);
    auto *dev_synapses = init_gpu_arr(synapses, 1);

    printf("Network: %d neurons (with segs: %d), %d synapses, %d generators\n",
           NRNS_NUMBER, NRNS_AND_SEGS, synapses_number, gens_number);

    int THREADS = 32, BLOCKS = 10;

    curandState *devStates;
    HANDLE_ERROR(cudaMalloc((void **)&devStates, NRNS_NUMBER * sizeof(curandState)));

    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // call initialisation kernel
    initialization_kernel<<<1, 1>>>(devStates, dev_S, dev_P, dev_N, -70.0);

    // the main simulation loop
    for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; ++sim_iter) {
        /// KERNEL ZONE
        // deliver_net_events, synapse updating and neuron conductance changing kernel
        synapse_kernel<<<5, 256>>>(dev_N, dev_synapses);
        // updating neurons kernel
        neuron_kernel<<<BLOCKS, THREADS>>>(devStates, dev_S, dev_P, dev_N, dev_G, sim_iter);
        /// SAVE DATA ZONE
        memcpyDtH(S->Vm, Vm, NRNS_AND_SEGS);
        memcpyDtH(N->g_exc, g_exc, NRNS_NUMBER);
        memcpyDtH(N->g_inh_A, g_inh_A, NRNS_NUMBER);
        memcpyDtH(N->g_inh_B, g_inh_B, NRNS_NUMBER);
        memcpyDtH(N->has_spike, has_spike, NRNS_NUMBER);
        // fill records arrays
        for (GroupMetadata& metadata : saving_groups) {
            copy_data_to(metadata, Vm, tmp, g_exc, g_inh_A, g_inh_B, has_spike, sim_iter);
        }
    }
    // properly ending work with GPU
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    // todo optimize the code to free all GPU variables
    HANDLE_ERROR(cudaFree(S->Vm));

    // stuff info
    printf("Elapsed GPU time: %d ms\n", (int) time);
    double Tbw = 12000 * pow(10, 6) * (128 / 8) * 2 / pow(10, 9);
    printf("Theoretical Bandwidth GPU (2 Ghz, 128 bit): %.2f GB/s\n", Tbw);
    // save the data into the current folder
    save_result(1);

}
void custom(void (*t_network)(), int steps = 1, int test_ind=1, int TEST = 0, int E2F_coef = 1, int V0v2F_coef = 1, int QUADRU_Ia = 1,
            string_code mode = quadru, string_code pharma = normal, string_code speed = s13){
    switch(speed) {
        case s6:
            skin_time = 125;
            break;
        case s13:
            skin_time = 50;
            break;
        case s21:
            skin_time = 25;
            break;
        default:
            exit(-1);
    }
    switch(mode) {
        case air:
            TEST = -1;
            skin_time = 25;
            cv_coef = 0.03; // 037
            E_coef = 0.04;
            slices_extensor = 5;
            slices_flexor = 4;
            E2F_coef = 0;
            V0v2F_coef = 0;
            break;
        case toe:
            TEST = -2;
            cv_coef = 0.035;
            E_coef = 0.04;
            slices_extensor = 4;
            slices_flexor = 4;
            E2F_coef = 8;
            V0v2F_coef = 0;
            break;
        case plt:
            cv_coef = 0.05; // 037
            E_coef = 0.04;
            slices_extensor = 6;
            slices_flexor = 5;
            E2F_coef = 10;
            V0v2F_coef = 0.001;
            break;
        case quadru:
            QUADRU_Ia = 0.6;
            cv_coef = 0.03; // 037
            E_coef = 0.03;
            slices_extensor = 6;
            slices_flexor = 7;
            E2F_coef = 10;
            V0v2F_coef = 0.001;
            break;
        default:
            exit(-1);
    }
    //
    switch(pharma) {
        case normal:
            break;
        case qpz:
            cv_coef = 0.12;
            E_coef = 0.08;
            V0v2F_coef = 0.01;
            break;
        case str:
            str_flag = true;
            V0v2F_coef = 0.01;
            break;
        default:
            exit(-1);
    }

    one_step_time = slices_extensor * skin_time + 25 * slices_flexor;
    sim_time = 25 + one_step_time * steps;
    SIM_TIME_IN_STEPS = (unsigned int)(sim_time / dt);  // [steps] converted time into steps

    // init the device
    int dev = 0;
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s \n", dev, deviceProp.name);
    HANDLE_ERROR(cudaSetDevice(dev));
    // the main body of simulation
    simulate(t_network);
    // reset device
    HANDLE_ERROR(cudaDeviceReset());
}