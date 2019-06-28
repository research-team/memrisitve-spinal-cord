#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __global__
	inline void __syncthreads() {}
#endif

#define DEBUG
#define COLOR_RED "\x1b[1;31m"
#define COLOR_GREEN "\x1b[1;32m"
#define COLOR_RESET "\x1b[0m"
#include <iostream>   // output/input streams definitions
#include <unistd.h>   // UNIX standard definitions
#include <fcntl.h>    // file control definitions
#include <termios.h>  // POSIX terminal control definitions
#include <cstring>    // string function definitions


#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>
#include <cstdlib>
#include <sys/io.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <ctime>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include "Group.cpp"
#include "SynapseMetadata.cpp"

using namespace std;

constexpr const char* const SERIAL_PORT_USB = "/dev/ttyUSB0";

void prepare_UART();
void close_UART();

bool gpu_is_run = true;         // special flag for controlling ADC/DAC working
bool EES_signal = false;        // special flag for signalling EES
float motoneuron_voltage = 0;   // common global variable of motoneuron membrane potential

unsigned int global_id = 0;
const unsigned int syn_outdegree = 27;
const unsigned int neurons_in_ip = 196;
const unsigned int neurons_in_moto = 169;
const unsigned int neurons_in_group = 20;
const unsigned int neurons_in_afferent = 196;

// simulation paramters
const float T_sim = 1000;
// 6 cm/s = 125 [ms]
// 15 cm/s = 50 [ms]
// 21 cm/s = 25 [ms]
const int skin_stim_time = 25;
const float INH_COEF = 1.0f;
const float SIM_STEP = 0.25;
const unsigned int sim_time_in_step = (unsigned int)(T_sim / SIM_STEP);

// neuron parameters
const float C = 100;        // [pF] membrane capacitance
const float V_rest = -72;   // [mV] resting membrane potential
const float V_thld = -55;   // [mV] spike threshold
const float k = 0.7;        // [pA * mV-1] constant ("1/R")
const float a = 0.02;       // [ms-1] time scale of the recovery variable U_m. Higher a, the quicker recovery
const float b = 0.2;        // [pA * mV-1] sensitivity of U_m to the sub-threshold fluctuations of the V_m
const float c = -80;        // [mV] after-spike reset value of V_m
const float d = 6;          // [pA] after-spike reset value of U_m
const float V_peak = 35;    // [mV] spike cutoff value
// [step] time of C0/C1 activation (for generators and calculating swapping)
const unsigned int steps_activation_C0 = (unsigned int)(5 * skin_stim_time / SIM_STEP);
const unsigned int steps_activation_C1 = (unsigned int)(6 * skin_stim_time / SIM_STEP);

// set timing
const auto frame_time = chrono::microseconds(150);
const auto dac_sleep_time = chrono::microseconds(50);
const auto adc_sleep_time = chrono::microseconds(50);

int serial_port;

// device variables

pthread_t thread_DAC;

// global vectors of SynapseMetadata of synapses for each neuron
vector<vector<SynapseMetadata>> metadatas;

#ifdef DEBUG
short ADC_data[sim_time_in_step];
short DAC_data[sim_time_in_step];
float normalized[sim_time_in_step];
float motodata[sim_time_in_step];
#endif

void errorchk(bool condition, string text) {
	cout << text << " ... ";
	if (condition) {
		cout << COLOR_RED "ERROR" COLOR_RESET << endl;
		cout << COLOR_RED "FAILED!" COLOR_RESET << endl;
		close_UART();
		exit(0);
	} else {
		cout << COLOR_GREEN "OK" COLOR_RESET << endl;
	}
}

// structure to store the port settings in
struct termios port_settings;
// struct for terminal app, since terminal also connects through a virtual system serial port
// "/dev/ttyUSB0" so it is necessary to redirect input and output, however this wouldn't be necessary
// if we want to send data streams from the program directly and also if we don't need to show the
// raw output to the user.
struct termios stdio;
// after our work is over we can reset the terminal input and output to the os instead of to and from the serial port
struct termios old_port_settings;

void prepare_UART() {
	// get the current options of the STDOUT_FILENO
	tcgetattr(STDOUT_FILENO, &old_port_settings);

	printf("Start with %s \n", SERIAL_PORT_USB);

	// O_RDWR - read/write access to serial port
	// O_NONBLOCK - when possible, the file is opened in nonblocking mode
	serial_port = open(SERIAL_PORT_USB, O_RDWR | O_NONBLOCK);
	errorchk(serial_port < 0, "open serial port");

	// populate the structures with the memory size of the structure by zeroes
	memset(&stdio, 0, sizeof(stdio));
	memset(&port_settings, 0, sizeof(port_settings));

	// output parameters
	stdio.c_iflag = 0;
	stdio.c_oflag = 0;
	stdio.c_cflag = 0;
	stdio.c_lflag = 0;
	stdio.c_cc[VMIN] = 0;
	stdio.c_cc[VTIME] = 0;

	// set the new options for the port
	// TCSANOW - make changes now without waiting for data to complete
	// TCSAFLUSH - flush input and output buffers and make the change
	tcsetattr(STDOUT_FILENO, TCSANOW, &stdio);
	tcsetattr(STDOUT_FILENO, TCSAFLUSH, &stdio);

	// make the reads non-blocking
	fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

	// input mode flags:
	port_settings.c_iflag = 0;
	// output mode flags: 0 - raw output (no output processing)
	port_settings.c_oflag = 0;
	// local mode flags:
	port_settings.c_lflag = 0;
	// control mode flags: 8 bits, no parity, 1 stop bit | enable receiver | ignore modem control lines
	port_settings.c_cflag |= (CS8 | CREAD | CLOCAL);
	// minimum number of characters to read => 0 - read doesn't block
	port_settings.c_cc[VMIN] = 0;
	// time to wait for data (tenths of seconds) => 0 seconds read timeout
	port_settings.c_cc[VTIME] = 0;

	cfsetispeed(&port_settings, B115200);  // set read speed as 115 200 (bps)
	cfsetospeed(&port_settings, B115200);  // set write speed as 115 200 (bps)

	// set the new options for the serial port
	tcsetattr(serial_port, TCSANOW, &port_settings);

}

void close_UART() {
	// close the serial port
	close(serial_port);

	// restore old port settings
	tcsetattr(STDOUT_FILENO, TCSANOW, &old_port_settings);
}

void *parallel_DAC_UART(void *arg) {
	unsigned char data[2];
	unsigned short kek;

	while (gpu_is_run) {
		kek = (unsigned short) motoneuron_voltage;
		printf("mot: %7.2f (%7.2f) -> kek: %d -> \n", motoneuron_voltage / 10, motoneuron_voltage, kek);

		if (kek > 0) {
			kek = kek + 2048;
		} else {
			kek = 2048;
		}

		data[0] = kek >> 0x08; //kek * 10 + 2048;
		data[1] = kek & 0xFF; //kek * 10 + 2048;
		write(serial_port, data, 2);
		this_thread::sleep_for(dac_sleep_time);
	}

	errorchk(false, "finishing DAC executing commands");

	return 0;
}


Group form_group(string group_name, unsigned int nrns_in_group = neurons_in_group) {
	// form structs of neurons global ID and groups name
	Group group = Group();

	group.group_name = group_name;
	group.id_start = global_id;
	group.id_end = global_id + nrns_in_group - 1;
	group.group_size = nrns_in_group;

	global_id += nrns_in_group;

	for (int i = 0; i < nrns_in_group; i++){
		metadatas.push_back(vector<SynapseMetadata>());
	}

	printf("Formed %s IDs [%d ... %d] = %d\n",
	       group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

__host__
int ms_to_step(float ms) { return (int)(ms / SIM_STEP); }

__global__
void gpu_neuron_kernel(float* old_v,
                       float* old_u,
                       int neurons_number,
                       float* nrn_current,
                       int* nrn_ref_time,
                       int* nrn_ref_time_timer,
                       bool* has_spike,
                       int activated_C_,
                       int shift_time_by_step,
                       float* gpu_motoneuron_voltage,
                       #ifdef DEBUG
                       float* voltage,
                       #endif
                       int sim_iter,
                       int t_cutting_offset,
                       bool gpu_EES_signal) {

	// get id of the thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0) {
		*gpu_motoneuron_voltage = 0;
	}

	__syncthreads();

	if (tid < neurons_number) {
		// reset spike flag of the current neuron
		has_spike[tid] = false;

		if (activated_C_ == 0){
			if (996 <= tid && tid <= 1164) {
				atomicAdd(gpu_motoneuron_voltage, V_rest);
			}
			return;
		}
		// C1
		if (0 <= tid && tid <= 19
			&& shift_time_by_step <= sim_iter
			&& sim_iter < (100 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000; // enough for spike
		}
		// C2
		if (20 <= tid && tid <= 39
			&& (100 + shift_time_by_step) <= sim_iter
			&& sim_iter < (200 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000; // enough for spike
		}
		// C3
		if (40 <= tid && tid <= 59
			&& (200 + shift_time_by_step) <= sim_iter
			&& sim_iter < (300 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000;
		}
		// C4
		if (60 <= tid && tid <= 79
			&& (300 + shift_time_by_step) <= sim_iter
			&& sim_iter < (500 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000;
		}
		// C5
		if (80 <= tid && tid <= 99
			&& (500 + shift_time_by_step) <= sim_iter
			&& sim_iter < (600 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000;
		}

		if (nrn_ref_time_timer[tid] > 0)
			nrn_current[tid] = 0;

		// EES
		if (sim_iter % 100 == 0 && 1165 <= tid && tid <= 1184) { // gpu_EES_signal &&
			nrn_current[tid] = 5000;
		}

		float V_old = old_v[tid];
		float U_old = old_u[tid];
		float I_current = nrn_current[tid];

		if (I_current > 10000)
			I_current = 10000;
		if (I_current < -10000)
			I_current = -10000;

		// re-calculate V_m and U_m
		float V_m = V_old + SIM_STEP * (k * (V_old - V_rest) * (V_old - V_thld) - U_old + I_current) / C;
		float U_m = U_old + SIM_STEP * a * (b * (V_old - V_rest) - U_old);

		// set bottom border of the membrane potential
		if (V_m < c)
			V_m = c;
		// set top border of the membrane potential
		if (V_m >= V_thld)
			V_m = V_peak;

		if (996 <= tid && tid <= 1164 && t_cutting_offset == 0 && V_m > -30) {
			V_m = -30;
		}

		#ifdef DEBUG
		int index = sim_iter + tid * sim_time_in_step;
		voltage[index] = V_m;
		#endif

		if (996 <= tid && tid <= 1164) {
			atomicAdd(gpu_motoneuron_voltage, V_m);
		}

		// threshold crossing (spike)
		if (V_m >= V_thld) {
			// set spike status
			has_spike[tid] = true;
			// redefine V_old and U_old
			old_v[tid] = c;
			old_u[tid] += d;
			// set the refractory period
			nrn_ref_time_timer[tid] = nrn_ref_time[tid];
		} else {
			// redefine V_old and U_old
			old_v[tid] = V_m;
			old_u[tid] = U_m;
		}
	}
}

__global__
void gpu_synapse_kernel(bool* has_spike,
                        float* nrn_current,
                        int *synapses_number,
                        int **synapses_post_nrn_id,
                        int **synapses_delay,
                        int **synapses_delay_timer,
                        float **synapses_weight,
                        int neurons_number,
                        int* nrn_ref_time_timer){

	// get ID of the thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// ignore threads which ID is greater than neurons number
	if (tid < neurons_number) {
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

		// update currents of the neuron ToDo it is a hotfix (must be in neuron kernel)
		float I_current = nrn_current[tid];
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
	}
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons, float syn_delay, float weight, int outdegree = syn_outdegree) {
	// connect neurons with uniform distribution and normal distributon for syn delay and weight
	weight *= (100 * 0.7);
	random_device rd;
	mt19937 gen(rd());  // Initialize pseudo-random number generator

	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	normal_distribution<float> weight_distr(weight, 2);
	normal_distribution<float> delay_distr(syn_delay, 0.1);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (unsigned int i = 0; i < outdegree; i++) {
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

unsigned int init_network() {
	// Form neuron groups
	Group C1 = form_group("C1");
	Group C2 = form_group("C2");
	Group C3 = form_group("C3");
	Group C4 = form_group("C4");
	Group C5 = form_group("C5");

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
	Group MP_E = form_group("MP_E", neurons_in_moto);
	Group EES = form_group("EES");
	Group Ia = form_group("Ia", neurons_in_afferent);

	Group inh_group3 = form_group("inh_group3");
	Group inh_group4 = form_group("inh_group4");
	Group inh_group5 = form_group("inh_group5");

	Group ees_group1 = form_group("ees_group1");
	Group ees_group2 = form_group("ees_group2");
	Group ees_group3 = form_group("ees_group3");
	Group ees_group4 = form_group("ees_group4");

	// set conenctions
	connect_fixed_outdegree(C3, inh_group3, 0.5, 15.0);
	connect_fixed_outdegree(C4, inh_group4, 0.5, 15.0);
	connect_fixed_outdegree(C5, inh_group5, 0.5, 15.0);

	connect_fixed_outdegree(inh_group3, G1_3, 2.8, 20.0);

	connect_fixed_outdegree(inh_group4, G1_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group4, G2_3, 1.0, 20.0);

	connect_fixed_outdegree(inh_group5, G1_3, 2.0, 20.0);
	connect_fixed_outdegree(inh_group5, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G4_3, 1.0, 20.0);

	/// D1
	// input from sensory
	connect_fixed_outdegree(C1, D1_1, 1, 0.4);
	connect_fixed_outdegree(C1, D1_4, 1, 0.4);
	connect_fixed_outdegree(C2, D1_1, 1, 0.4);
	connect_fixed_outdegree(C2, D1_4, 1, 0.4);
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
	connect_fixed_outdegree(C2, D2_1, 1, 0.8);
	connect_fixed_outdegree(C2, D2_4, 1, 0.8);
	connect_fixed_outdegree(C3, D2_1, 1, 0.8);
	connect_fixed_outdegree(C3, D2_4, 1, 0.8);
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
	connect_fixed_outdegree(C3, D3_1, 1, 0.5);
	connect_fixed_outdegree(C3, D3_4, 1, 0.5);
	connect_fixed_outdegree(C4, D3_1, 1, 0.5);
	connect_fixed_outdegree(C4, D3_4, 1, 0.5);
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
	connect_fixed_outdegree(C4, D4_1, 1, 0.7); // was 0.5
	connect_fixed_outdegree(C4, D4_4, 1, 0.7); // was 0.5
	connect_fixed_outdegree(C5, D4_1, 1, 0.7); // was 0.5
	connect_fixed_outdegree(C5, D4_4, 1, 0.7); // was 0.5
	// input from Group (3)
	connect_fixed_outdegree(ees_group3, D4_1, 1, 1.5); // was 1.2
	connect_fixed_outdegree(ees_group3, D4_4, 1, 1.5); // was 1.2
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
	connect_fixed_outdegree(C5, D5_1, 1, 0.5);
	connect_fixed_outdegree(C5, D5_4, 1, 0.5);
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
	connect_fixed_outdegree(G1_2, IP_E, 3, 25.0);

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
	connect_fixed_outdegree(G3_2, IP_E, 2, 25.0);

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
	connect_fixed_outdegree(G4_2, IP_E, 1.0, 17.0);

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
	connect_fixed_outdegree(G5_2, IP_E, 2, 20.0);

	connect_fixed_outdegree(IP_E, MP_E, 1, 11);
	connect_fixed_outdegree(EES, MP_E, 2, 50);
	connect_fixed_outdegree(Ia, MP_E, 1, 1);

	return static_cast<unsigned int>(metadatas.size());
}

#ifdef DEBUG
void save_ADC_DAC_mean(unsigned int data_index) {
	char cwd[256];
	ofstream myfile;

	getcwd(cwd, sizeof(cwd));
	printf("Save ADC results to: %s \n", cwd);

	string new_name = "/ADC.dat";
	myfile.open(cwd + new_name);
	for(int i = 0; i < data_index; i++)
		myfile << ADC_data[i] << " ";
	myfile << endl;
	for(int i = 0; i < data_index; i++)
		myfile << DAC_data[i] << " ";
	myfile << endl;
	for(int i = 0; i < data_index; i++)
		myfile << normalized[i] << " ";
	myfile << endl;
	for(int i = 0; i < data_index; i++)
		myfile << motodata[i] << " ";
	myfile.close();
}

void save_result(float* voltage_recording, int neurons_number) {
	// save results for each neuron (voltage/current/spikes)
	char cwd[256];
	ofstream myfile;

	getcwd(cwd, sizeof(cwd));
	printf("Save results to: %s \n", cwd);

	string new_name = "/volt.dat";
	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++)
			myfile << voltage_recording[sim_iter + nrn_id * sim_time_in_step] << " ";
		myfile << "\n";
	}

	myfile.close();
}
#endif

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
void simulate() {
	chrono::time_point<chrono::system_clock> iter_t_start, iter_t_end, simulation_t_start, simulation_t_end;

	// create neuron groups/synapses and get number of neurons
	const unsigned int neurons_number = init_network();

	// prepare number of threads
	int threads_per_block = 512;
	int num_blocks = neurons_number / threads_per_block + 1;

	// sim stuff variables
	int local_iter = 0;
	int activated_C_ = 0;
	int shift_time_by_step = 0;

	// init GPU pointers
	float* gpu_old_v;
	float* gpu_old_u;
	bool* gpu_has_spike;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_timer;
	float* gpu_nrn_current;
	int* gpu_synapses_number;
	float *gpu_motoneuron_voltage;
	// 2D pointers
	int **gpu_synapses_post_nrn_id;
	int **gpu_synapses_delay;
	int **gpu_synapses_delay_timer;
	float **gpu_synapses_weight;

	// init CPU variables (1D arrays)
	float old_v[neurons_number];
	float old_u[neurons_number];
	bool has_spike[neurons_number];
	int nrn_ref_time[neurons_number];
	int nrn_ref_timer[neurons_number];
	float nrn_current[neurons_number];
	int synapses_number[neurons_number];
	// allocate memory for pointers (2D arrays)
	int **synapses_post_nrn_id = (int **)malloc(datasize<int* >(neurons_number));
	int **synapses_delay = (int **)malloc(datasize<int* >(neurons_number));
	int **synapses_delay_timer = (int **)malloc(datasize<int* >(neurons_number));
	float **synapses_weight = (float **)malloc(datasize<float* >(neurons_number));

	// fill array by default values
	init_array<float>(old_v, neurons_number, V_rest);
	init_array<float>(old_u, neurons_number, 0);
	init_array<bool>(has_spike, neurons_number, false);
	init_array<int>(nrn_ref_time, neurons_number, ms_to_step(3.0));
	init_array<int>(nrn_ref_timer, neurons_number, -1);
	init_array<float>(nrn_current, neurons_number, 0);

	// fill arrays of synapses (per each neuron)
	for(unsigned int neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		// get number of synapses for current neuron
		unsigned int syn_count = static_cast<unsigned int>(metadatas.at(neuron_id).size());
		synapses_number[neuron_id] = syn_count;

		// prepare arrays for filling and copying to the GPU
		int tmp_synapses_post_nrn_id[syn_count];
		int tmp_synapses_delay[syn_count];
		int tmp_synapses_delay_timer[syn_count];
		float tmp_synapses_weight[syn_count];

		int syn_id = 0;
		// fill temporary arrays
		for(SynapseMetadata metadata : metadatas.at(neuron_id)) {
			tmp_synapses_post_nrn_id[syn_id] = metadata.post_id;
			tmp_synapses_delay[syn_id] = metadata.synapse_delay;
			tmp_synapses_delay_timer[syn_id] = -1;
			tmp_synapses_weight[syn_id] = metadata.synapse_weight;
			syn_id++;
		}

		// allocate memory on GPU
		cudaMalloc((void**)&synapses_post_nrn_id[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay_timer[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_weight[neuron_id], datasize<float>(syn_count));
		// copy data from temporary arrays on CPU to GPU
		cudaMemcpy(synapses_post_nrn_id[neuron_id], &tmp_synapses_post_nrn_id, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay[neuron_id], &tmp_synapses_delay, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay_timer[neuron_id], &tmp_synapses_delay_timer, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_weight[neuron_id], &tmp_synapses_weight, datasize<float>(syn_count), cudaMemcpyHostToDevice);
	}
	// allocate memory on GPU for one variable
	cudaMalloc((void**)&gpu_motoneuron_voltage, sizeof(float));
	// allocate memory on GPU for 1D arrays
	cudaMalloc(&gpu_old_v, datasize<float>(neurons_number));
	cudaMalloc(&gpu_old_u, datasize<float>(neurons_number));
	cudaMalloc(&gpu_has_spike, datasize<bool>(neurons_number));
	cudaMalloc(&gpu_nrn_current, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_timer, datasize<int>(neurons_number));
	cudaMalloc(&gpu_synapses_number, datasize<int>(neurons_number));
	// allocate memory on GPU for 2D arrays
	cudaMalloc((void ***)&gpu_synapses_post_nrn_id, datasize<int *>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_delay, datasize<int *>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_delay_timer, datasize<int *>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_weight, datasize<float *>(neurons_number));

	// copy data from CPU to GPU
	memcpyHtD<float>(gpu_old_v, old_v, neurons_number);
	memcpyHtD<float>(gpu_old_u, old_u, neurons_number);
	memcpyHtD<bool>(gpu_has_spike, has_spike, neurons_number);
	memcpyHtD<float>(gpu_nrn_current, nrn_current, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_timer, nrn_ref_timer, neurons_number);
	memcpyHtD<int>(gpu_synapses_number, synapses_number, neurons_number);
	memcpyHtD<int *>(gpu_synapses_post_nrn_id, synapses_post_nrn_id, neurons_number);
	memcpyHtD<int *>(gpu_synapses_delay, synapses_delay, neurons_number);
	memcpyHtD<int *>(gpu_synapses_delay_timer, synapses_delay_timer, neurons_number);
	memcpyHtD<float *>(gpu_synapses_weight, synapses_weight, neurons_number);
	cudaMemcpy(gpu_motoneuron_voltage, &motoneuron_voltage, sizeof(float), cudaMemcpyHostToDevice);

	#ifdef DEBUG
	float* gpu_voltage_recording;
	float* voltage_recording = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_step));
	init_array<float>(voltage_recording, neurons_number * sim_time_in_step, V_rest);
	cudaMalloc(&gpu_voltage_recording, datasize<float>(neurons_number * sim_time_in_step));
	memcpyHtD<float>(gpu_voltage_recording, voltage_recording, neurons_number * sim_time_in_step);
	#endif

	// create ADC thread
//	pthread_create(&thread_ADC, NULL, parallel_ADC_E154, 0);
//	errorchk(false, "start ADC parallel function");

	// create DAC thread
//	pthread_create(&thread_DAC, NULL, parallel_DAC_UART, 0);
//	errorchk(false, "start DAC parallel function");

	// show the GPU properties
	printf("GPU properties: %d threads x %d blocks (Total: %d threads mapped on %d neurons)\n",
	       threads_per_block, num_blocks, threads_per_block * num_blocks, neurons_number);
	errorchk(false, "start GPU");

	// start measure time for the simulation
	simulation_t_start = chrono::system_clock::now();
	#ifdef DEBUG
	unsigned int data_index = 0;
	#endif
	unsigned short t_cutting_offset = 0;

	unsigned char data[2];
	unsigned short kek;
	// sim iterations
	for (unsigned int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++) {
		// start measure time of one iteration
		iter_t_start = chrono::system_clock::now();

		if (sim_iter % 100 == 0) {
			t_cutting_offset = 30;
		}

		// if flexor C0 activated, find the end of it and change to C1
		if (activated_C_ == 0) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				activated_C_ = 1; // change to C1
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C0;  // add constant 125 ms
			}
			// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				activated_C_ = 0; // change to C0
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C1;  // add time equal to n_layers * 25 ms
			}
		}
		// run GPU kernel for neurons state updating
		gpu_neuron_kernel<<<num_blocks, threads_per_block>>>(gpu_old_v,
		                                                     gpu_old_u,
		                                                     neurons_number,
		                                                     gpu_nrn_current,
		                                                     gpu_nrn_ref_time,
		                                                     gpu_nrn_ref_timer,
		                                                     gpu_has_spike,
		                                                     activated_C_,
		                                                     shift_time_by_step,
		                                                     gpu_motoneuron_voltage,
		                                                     #ifdef DEBUG
		                                                     gpu_voltage_recording,
		                                                     #endif
		                                                     sim_iter,
		                                                     t_cutting_offset,
		                                                     EES_signal);
		// run GPU kernel for synapses state updating
		gpu_synapse_kernel<<<num_blocks, threads_per_block>>>(gpu_has_spike,
		                                                      gpu_nrn_current,
		                                                      gpu_synapses_number,
		                                                      gpu_synapses_post_nrn_id,
		                                                      gpu_synapses_delay,
		                                                      gpu_synapses_delay_timer,
		                                                      gpu_synapses_weight,
		                                                      neurons_number,
		                                                      gpu_nrn_ref_timer);

		if (EES_signal) {
			EES_signal = false;
		}
		// get summarized membrane potential value of motoneurons and normalize it to 5V peak of DAC
		cudaMemcpy(&motoneuron_voltage, gpu_motoneuron_voltage, sizeof(float), cudaMemcpyDeviceToHost);

		#ifdef DEBUG
		motodata[data_index] = motoneuron_voltage / neurons_in_moto;
//		normalized[data_index] = (motoneuron_voltage / neurons_in_moto - c) / 65 * 5;
//		DAC_data[data_index] = async_par_dac.Data[0];
//		ADC_data[data_index++] = data1[0];
		#endif
		motoneuron_voltage = (motoneuron_voltage / neurons_in_moto - c) * 10;

		kek = (unsigned short) motoneuron_voltage;
//		printf("mot: %7.2f (%7.2f) -> kek: %d -> \n", motoneuron_voltage / 10, motoneuron_voltage, kek);

		if (kek > 0) {
			kek = kek + 2048;
		} else {
			kek = 2048;
		}


		data[0] = kek >> 0x08; //kek * 10 + 2048;
		data[1] = kek & 0xFF; //kek * 10 + 2048;
		write(serial_port, data, 2);

		// stop measure time
		iter_t_end = std::chrono::system_clock::now();

		// get time difference
		auto elapsed = chrono::duration_cast<chrono::microseconds>(iter_t_end - iter_t_start);
		// wait if we are so fast
		if (elapsed < frame_time) {
			this_thread::sleep_for(frame_time - elapsed);
		}
		local_iter++;

		if (t_cutting_offset != 0)
			t_cutting_offset--;
	} // end of simulation iteration loop

	simulation_t_end = chrono::system_clock::now();

	// send "signal" to break the loop of ADC/DAC async functions
	gpu_is_run = false;

	// close threads properly
//	pthread_join(thread_ADC, NULL);
//	pthread_join(thread_DAC, NULL);

	// show timing information
	auto sim_time_diff = chrono::duration_cast<chrono::milliseconds>(simulation_t_end - simulation_t_start).count();
	printf("Elapsed %li ms (measured) | T_sim = %.2f ms\n", sim_time_diff, T_sim);
	printf("%s x%f\n", (double)(T_sim / sim_time_diff) > 1?
	                           COLOR_GREEN "faster" COLOR_RESET: COLOR_RED "slower" COLOR_RESET, T_sim / sim_time_diff);

	#ifdef DEBUG
	// copy neurons/synapses array to the HOST
	memcpyDtH<float>(voltage_recording, gpu_voltage_recording, neurons_number * sim_time_in_step);
	#endif

	// tell the CPU to halt further processing until the CUDA kernel has finished doing its business
	cudaDeviceSynchronize();
	// remove all device allocations
	cudaDeviceReset();

	#ifdef DEBUG
	save_ADC_DAC_mean(data_index);
	save_result(voltage_recording, neurons_number);
	#endif
}

int main() {
//	prepare_E154();
	prepare_UART();

	simulate();

//	close_E154();
	close_UART();

	// you are awesome
	cout << COLOR_GREEN "SUCCESS" COLOR_RESET << endl;

	return 0;
}