#include "core.cu"
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

Group form_group(const string &group_name,
                 int nrns_in_group = neurons_in_group,
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
