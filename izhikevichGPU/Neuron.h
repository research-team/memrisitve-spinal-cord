#ifndef IZHIKEVICHGPU_NEURON_H
#define IZHIKEVICHGPU_NEURON_H

//#include <openacc.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <utility>

using namespace std;

extern const float T_sim;

class Neuron {
	/// Synapse structure
	struct Synapse {
		Neuron* post_neuron{}; // post neuron ID
		int syn_delay{};		 // [steps] synaptic delay. Converts from ms to steps
		float weight{};		 // [pA] synaptic weight
		int timer{};			 // [steps] changeable value of synaptic delay
		float changing_weight{};

		Synapse() = default;
		Synapse(Neuron* post, float delay, float w) {
			this-> post_neuron = post;
			this-> syn_delay = ms_to_step(delay);
			this-> weight = w;
			this-> timer = -1;
			this-> changing_weight = w;
		}
	};
private:
	/// Object variables
	int id{};								// neuron ID
	float *spike_times{};					// array of spike time
	float *membrane_potential{};			// array of membrane potential values
	float *I_potential{};					// array of I

	int mm_record_step = ms_to_step(0.1f); 	// step of recording membrane potential
	int iterSpikesArray = 0;				// current index of array of the spikes
	int iterVoltageArray = 0; 				// current index of array of the V_m
	int simulation_iter = 0;		        // current simulation step
	bool hasMultimeter = false;				// if neuron has multimeter
	bool hasSpikedetector = false;			// if neuron has spikedetector
	bool hasGenerator = false;				// if neuron has generator

	/// Stuff variables
	const float I_tau = 3.0f;				                            // step of I decreasing/increasing
	static constexpr float ms_in_1step = 0.1f;	                        // how much milliseconds in 1 step
	static const int steps_in_1ms = static_cast<int>(1 / ms_in_1step);  // how much steps in 1 ms

	/// Parameters (const)
	const float C = 100.0f;			// [pF] membrane capacitance
	const float V_rest = -72.0f;	// [mV] resting membrane potential
	const float V_th = -55.0f;		// [mV] spike threshold
	const float k = 0.7f;			// [pA * mV-1] constant ("1/R")
	const float a = 0.03f;			// [ms-1] time scale of the recovery variable U_m
	const float b = -2.0f;			// [pA * mV-1]  sensitivity of U_m to the sub-threshold fluctuations of the V_m
	const float c = -80.0f;			// [mV] after-spike reset value of V_m
	const float d = 100.0f;			// [pA] after-spike reset value of U_m
	const float V_peak = 35.0f;		// [mV] spike cutoff value
	int ref_t{}; 					// [step] refractory period

	/// State (changable)
	float V_m = V_rest;		// [mV] membrane potential
	float U_m = 0.0f;		// [pA] membrane potential recovery variable
	float I = 0.0f;			// [pA] input current
	float V_old = V_m;		// [mV] previous value for the V_m
	float U_old = U_m;		// [pA] previous value for the U_m
	float current_ref_t = 0;

public:
	Synapse* synapses = new Synapse[100];	// array of synapses
	int num_synapses{0};                    // current number of synapses (neighbors)
	char* name{};

	Neuron() = default;

	Neuron(int id, float ref_t) {
		this->id = id;
		this->ref_t = ms_to_step(ref_t);
	}

	void changeCurrent(float I) {
		if (!hasGenerator && this->I <= 400 && this->I >= -400) {
			this->I += I;
		}
	}

	void addMultimeter() {
		// set flag that this neuron has the multimeter
		hasMultimeter = true;
		// allocate memory for recording V_m
		membrane_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];
		// allocate memory for recording I
		I_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];

	}

	void addSpikedetector() {
		// set flag that this neuron has the multimeter
		hasSpikedetector = true;
		// allocate memory for recording spikes
		spike_times = new float[ ms_to_step(T_sim) / this->ref_t ];
	}

	void addGenerator(float I) {
		hasGenerator = true;
		this->I = I;
	}

	bool withMultimeter() {
		return hasMultimeter;
	}

	bool withSpikedetector() {
		return hasSpikedetector;
	}

	float step_to_ms(int step) {
		return step * ms_in_1step;  // convert steps to milliseconds
	}

	static int ms_to_step(float ms) {
		return (int) (ms * steps_in_1ms);   // convert milliseconds to step
	}

	Neuron* getThis() {
		return this;
	}

	char* getName() {
		return this->name;
	}

	int getID() {
		return this->id;
	}

	float* getSpikes() {
		return spike_times;
	}

	float* getVoltage() {
		return membrane_potential;
	}

	float* getCurrents() {
		return I_potential;
	}

	int getVoltageArraySize() {
		return (ms_to_step(T_sim) / mm_record_step);
	}

	int getSimulationIter() {
		return simulation_iter;
	}

	int getIterSpikesArray() {
		return iterSpikesArray;
	}

	//#pragma acc routine vector
	/// Invoked every simulation step, update the neuron state
	void update_state() {
		if (current_ref_t > 0) {
			// calculate V_m and U_m WITHOUT synaptic weight
			// (absolute refractory period)
			V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old) / C;
			U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);

		} else {
			// calculate V_m and U_m WITH synaptic weight
			// (action potential)
			V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old + I) / C;
			U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);
		}

		// save the V_m and I value every mm_record_step if hasMultimeter
		if (hasMultimeter && simulation_iter % mm_record_step == 0) {
			membrane_potential[iterVoltageArray] = V_m;
			I_potential[iterVoltageArray] = I;
			iterVoltageArray++;
		}

		if (V_m < c)
			V_m = c;

		// threshold crossing (spike)
		if (V_m >= V_peak) {
			// set timers for all neuron synapses
			for (int i = 0; i < num_synapses; i++) {
				synapses[i].timer = synapses[i].syn_delay;
			}

			// redefine V_old and U_old
			V_old = c;
			U_old += d;

			// save spike time if hasSpikedetector
			if (hasSpikedetector) {
				spike_times[iterSpikesArray] = step_to_ms(simulation_iter);
				iterSpikesArray++;
			}

			// set the refractory period
			current_ref_t = ref_t;
		} else {
			// redefine V_old and U_old
			V_old = V_m;
			U_old = U_m;
		}

		// update timers in all neuron synapses
		for (int i = 0; i < num_synapses; i++ ) {
			Synapse* syn = synapses + i;
			// "send spike"
			if (syn->timer == 0) {
				syn->post_neuron->changeCurrent(syn->weight);
				// set timer to -1 (thats mean no need to update timer in future without spikes)
				syn->timer = -1;
			}
			// decrement timers
			if (syn->timer > 0) {
				syn->timer--;
			}
		}

		// update I (currents) of the neuron
		// doesn't change the I of generator neurons!!!
		if (!hasGenerator && I != 0) {
			if (I > 0) {	// turn the current to 0 by I_tau step
				I -= I_tau;	// decrease I
				if (I < 0)	// avoid the near value to 0
					I = 0;
			} else {
				I += I_tau; // increase I
				if (I > 0)	// avoid the near value to 0
					I = 0;
			}
		}

		// update the refractory period timer
		if (current_ref_t > 0)
			current_ref_t--;

		// update the simulation iteration
		simulation_iter++;
	}

	void connectWith(Neuron* pre_neuron, Neuron* post_neuron, float syn_delay, float weight) {
		/// adding the new synapse to the neuron
		Synapse* syn = new Synapse(post_neuron, syn_delay, weight);
		pre_neuron->synapses[pre_neuron->num_synapses++] = *syn;
	}


	~Neuron() {
		//#pragma acc exit data delete(this)
		if (hasSpikedetector)
			delete[] spike_times;

		if (hasMultimeter) {
			delete[] membrane_potential;
			delete[] I_potential;
		}
		delete[] synapses;
	}
};

#endif //IZHIKEVICHGPU_NEURON_H
