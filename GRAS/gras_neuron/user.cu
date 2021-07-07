#include "core.cu"
#include "structs.h"

int TEST;
double E2F_coef;
double V0v2F_coef;
double QUADRU_Ia;
void init_network()
{
	/**
	 * todo
	 */
//	string name;
//	vector<Group> E, CV, L0, L1, L2E, L2F, L3, IP_E, IP_F, gen_C, C_0, V0v;
	// generators
//	auto ees = form_group("EES", 1, GENERATOR);
//	for(int layer = 0; layer < layers + 1; ++layer) {
//		name = to_string(layer + 1);
//		gen_C.push_back(form_group("C" + name, 1, GENERATOR));
//	}
//	for(int step = 0; step < step_number; ++step) {
//		name = to_string(step);
//		C_0.push_back(form_group("C_0_step_" + name, 1, GENERATOR));
//		V0v.push_back(form_group("V0v_step_" + name, 1, GENERATOR));
//
//	}
//	//
//	auto OM1_0E = form_group("OM1_0E");
//	auto OM1_0F = form_group("OM1_0F");
//	// OM groups by layer
//	for(int layer = 0; layer < layers; ++layer) {
//		name = to_string(layer + 1);
//		L0.push_back(form_group("OM" + name + "_0"));
//		L1.push_back(form_group("OM" + name + "_1"));
//		L2E.push_back(form_group("OM" + name + "_2E"));
//		L2F.push_back(form_group("OM" + name + "_2F"));
//		L3.push_back(form_group("OM" + name + "_3"));
////        T1.push_back(form_group("T1"));
////        T2.push_back(form_group("T2"));
////        T3.push_back(form_group("T3"));
//
//
//	}
//	//
//	for(int layer = 0; layer < layers + 1; ++layer) {
//		name = to_string(layer + 1);
//		E.push_back(form_group("E" + name, 50, AFFERENTS));
//		CV.push_back(form_group("CV_" + name, 50, AFFERENTS));
//		// interneuronal pool
//		IP_E.push_back(form_group("IP_E_" + name));
//		IP_F.push_back(form_group("IP_F_" + name));
//	}
//	// afferents
//	auto Ia_aff_E = form_group("Ia_aff_E", 120, AFFERENTS);
//	auto Ia_aff_F = form_group("Ia_aff_F", 120, AFFERENTS);
//	// motoneurons
//	auto mns_E = form_group("mns_E", 210, MOTO);
//	auto mns_F = form_group("mns_F", 180, MOTO);
//	// muscle fibers
//	auto muscle_E = form_group("muscle_E", 210 * 50, MUSCLE, 3); // 150 * 210
//	auto muscle_F = form_group("muscle_F", 180 * 50, MUSCLE, 3); // 100 * 180
//	// reflex arc E
//	auto Ia_E = form_group("Ia_E", neurons_in_ip);
//	auto iIP_E = form_group("iIP_E", neurons_in_ip);
//	auto R_E = form_group("R_E");
//	// reflex arc F
//	auto Ia_F = form_group("Ia_F", neurons_in_ip);
//	auto iIP_F = form_group("iIP_F", neurons_in_ip);
//	auto R_F = form_group("R_F");
//
////    auto TE = form_group("TE", GENERATOR);
//
//
//	// create EES generator
//	add_generator(ees, 0, sim_time, ees_fr);
//	// create CV generators (per step)
//	for (int layer = 0; layer < layers + 1 + TEST; ++layer) {
//		for (int step_index = 0; step_index < step_number; ++step_index) {
//			normal_distribution<double> freq_distr(cv_fr, cv_fr / 10);
//			double start = 25 + skin_time * layer + step_index * (skin_time * slices_extensor + 25 * slices_flexor);
//			double end = start + skin_time - 3; // remove merging CV
//			add_generator(gen_C[layer], start, end, freq_distr(rand_gen));
//		}
//		printf("step\n");
//	}
//	// create C_0 and V0v generators (per step)
//	for (int step_index = 0; step_index < step_number; ++step_index) {
//		// freq = 200 (interval = 5ms), count = 125 / interval. Duration = count * interval = 125
//		double start = 25 + skin_time * slices_extensor + step_index * (skin_time * slices_extensor + 25 * slices_flexor);
//		double end = start + 25 * slices_flexor;
//		add_generator(C_0[step_index], start, end, cv_fr);
//		// V0v
//		start = 20 + skin_time * slices_extensor + step_index * (skin_time * slices_extensor + 25 * slices_flexor);
//		end = start + 75; // 75
//		add_generator(V0v[step_index], start, end, cv_fr);
//	}
//
//	// extensor
//	createmotif(OM1_0E, L1[0], L2E[0], L3[0]);
//	for(int layer = 1; layer < layers; ++layer)
//		createmotif(L0[layer], L1[layer], L2E[layer], L3[layer]);
//	// extra flexor connections
//	createmotif_flexor(OM1_0F, L1[0], L2F[0], L3[0]);
//	for(int layer = 1; layer < layers; ++layer)
//		createmotif_flexor(L0[layer], L1[layer], L2F[layer], L3[layer]);
//
//	for(int layer = 1; layer < layers; ++layer)
//		connect_fixed_indegree(L2F[layer - 1], L2F[layer], 2, 0.5, 50, 2);
//	//
//
//	connect_fixed_indegree(E[0], OM1_0F, 3, 0.00025 * E2F_coef, 50, 3);
//	for(int step = 0; step < step_number; ++step) {
//		connect_fixed_indegree(V0v[step], OM1_0F, 3, 2.75 * V0v2F_coef, 50, 5);
//	}
//	// between delays via excitatory pools
//	// extensor
//	for(int layer = 1; layer < layers; ++layer) {
//		connect_fixed_indegree(E[layer - 1], E[layer], 3, 0.75); // 4.75
//	}
//	// connect E (from EES)
//	connect_fixed_indegree(E[0], OM1_0E, 2, 0.005 * 0.8 * E_coef, 50, 3); // 0.00040 - 0.00047
//	for(int layer = 1; layer < layers; ++layer) {
//		connect_fixed_indegree(E[layer], L0[layer], 2, 0.005 * 0.8 * E_coef, 50, 4); // 0.00048 * 0.4, 1.115
//	}
//
//	// E inhibitory projections (via 3rd core)
//	for (int layer = 0; layer < layers - 1; ++layer) {
//		if (layer >= 3) {
//			for (int i = layer + 3; i < layers + 1 + TEST; ++i) {
//				printf("C index %d, OM%d_3 (layer > 3)\n", i, layer);
//				connect_fixed_indegree(gen_C[i], L3[layer], 0.1, 0.1);
//			}
//		} else {
//			for (int i = layer + 2; i < layers + 1 + TEST; ++i) {
//				printf("C index %d, OM%d_3 (else)\n", i, layer);
//				connect_fixed_indegree(gen_C[i], L3[layer], 0.1, 0.1);
//			}
//		}
//	}
//	conn_generator(ees, Ia_aff_E, 1, 2.5);
//	conn_generator(ees, Ia_aff_F, 1, 2.5);
//	conn_generator(ees, E[0], 2, 1.0); // NORMAL
//	///conn_generator(Iagener_E, Ia_aff_E, 1, 0.0001, 5);
//	///conn_generator(Iagener_F, Ia_aff_F, 1, 0.0001, 5);
//
//	connect_fixed_indegree(Ia_aff_E, mns_E, 1, 0.1 * QUADRU_Ia); // was 1.5ms
//	connect_fixed_indegree(Ia_aff_F, mns_F, 2.5, 0.1);
//
//	connect_fixed_outdegree_MUSCLE(mns_E, muscle_E, 1.2, 0.11, 45); // 2.0
//	connect_fixed_outdegree_MUSCLE(mns_F, muscle_F, 1, 0.11, 45, 5); // 2.0
//
//	// IP
//	for (int layer = 0; layer < layers; ++layer) {
//		// Extensor
////		connectinsidenucleus(IP_F[layer]);
////		connectinsidenucleus(IP_E[layer]);
////		connectinsidenucleus(L2E[layer]);
////		connectinsidenucleus(L2F[layer]);
//		connect_fixed_indegree(L2E[layer], IP_E[layer], 2, 0.005, 500, 5); // 2.5
//		connect_fixed_indegree(IP_E[layer], mns_E, 2, 0.005, 500, 5); // 2.75 0.125 0.2
//		if (layer > 3)
//			connect_fixed_indegree(IP_E[layer], Ia_aff_E, 1, -layer * 0.0002);
//		else
//			connect_fixed_indegree(IP_E[layer], Ia_aff_E, 1, -0.0001);
//		// Flexor
//		connect_fixed_indegree(L2F[layer], IP_F[layer], 1, 0.001, 500, 5); // 2.5
//		connect_fixed_indegree(IP_F[layer], mns_F, 1, 0.003, 500, 5); // 2.75 0.125 0.2
//		connect_fixed_indegree(IP_F[layer], Ia_aff_F, 1, -0.95);
//	}
//	// skin inputs
//	for (int layer = 0; layer < layers + 1 + TEST; ++layer)
//		connect_fixed_indegree(gen_C[layer], CV[layer], 2, 0.15 * cv_coef);
//	// CV
    double TESTCOEF = 35.0; // 4.25
//	// OM1
//	connect_fixed_indegree(CV[0], OM1_0E, 2, 0.00075 * cv_coef * TESTCOEF * 0.2, 50, 3);
//	connect_fixed_indegree(CV[1], OM1_0E, 2, 0.00051 * cv_coef * TESTCOEF * 0.4, 50, 3);
//
//	// OM2
//	connect_fixed_indegree(CV[0], L0[1], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	connect_fixed_indegree(CV[1], L0[1], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	connect_fixed_indegree(CV[2], L0[1], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	// OM3
//	connect_fixed_indegree(CV[0], L0[2], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	connect_fixed_indegree(CV[1], L0[2], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	connect_fixed_indegree(CV[2], L0[2], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	connect_fixed_indegree(CV[3], L0[2], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	connect_fixed_indegree(CV[4], L0[2], 3, 0.00025 * cv_coef * TESTCOEF * 1.0, 50, 3);
//	// OM4
//	connect_fixed_indegree(CV[1], L0[3], 3, 0.00002 * cv_coef * TESTCOEF * 0.9, 50, 3);
//	connect_fixed_indegree(CV[2], L0[3], 3, 0.00022 * cv_coef * TESTCOEF * 0.9, 50, 3);
//	connect_fixed_indegree(CV[3], L0[3], 3, 0.00025 * cv_coef * TESTCOEF * 0.9, 50, 3);
//	connect_fixed_indegree(CV[4], L0[3], 3, 0.00025 * cv_coef * TESTCOEF * 0.9, 50, 3);
//	connect_fixed_indegree(CV[5], L0[3], 3, 0.00015 * cv_coef * TESTCOEF * 0.9, 50, 3);
//	// OM5
//	connect_fixed_indegree(CV[1], L0[4], 3, 0.00012 * cv_coef * TESTCOEF * 1.3, 50, 3);
//	connect_fixed_indegree(CV[2], L0[4], 3, 0.00012 * cv_coef * TESTCOEF * 1.3, 50, 3);
//	connect_fixed_indegree(CV[3], L0[4], 3, 0.00012 * cv_coef * TESTCOEF * 1.3, 50, 3);
//	connect_fixed_indegree(CV[4], L0[4], 3, 0.00012 * cv_coef * TESTCOEF * 1.3, 50, 3);
//	connect_fixed_indegree(CV[5], L0[4], 3, 0.00012 * cv_coef * TESTCOEF * 1.3, 50, 3);
//
//
//
//	// C=1 Extensor
//	for (int layer = 0; layer < layers; ++layer)
//		connect_fixed_indegree(IP_E[layer], iIP_E, 1, 0.001);
////	//
//	for (int layer = 0; layer < layers + 1 + TEST; ++layer) {
//		connect_fixed_indegree(CV[layer], iIP_E, 1, 1.8);
//		connect_fixed_indegree(gen_C[layer], iIP_E, 1, 1.8);
//	}
//	connect_fixed_indegree(iIP_E, OM1_0F, 0.1, -0.001);
//
//	for (int layer = 0; layer < layers; ++layer) {
//		connect_fixed_indegree(iIP_E, L2F[layer], 2, -1.8);
//		connect_fixed_indegree(iIP_F, L2E[layer], 2, -0.5);
//	}
//	//
//	connect_fixed_indegree(iIP_E, Ia_aff_F, 1, -1.2);
//	connect_fixed_indegree(iIP_E, mns_F, 0.1, -0.3);
//	for (int layer = 0; layer < layers; ++layer) {
//		connect_fixed_indegree(iIP_E, IP_F[layer], 1, -0.1);
//		connect_fixed_indegree(IP_F[layer], iIP_F, 1, 0.0001);
//		connect_fixed_indegree(iIP_F, IP_E[layer], 1, -0.5);
//	}
//	// C=0 Flexor
//	connect_fixed_indegree(iIP_F, iIP_E, 1, -0.5);
//	connect_fixed_indegree(iIP_F, Ia_aff_E, 1, -0.5);
//	connect_fixed_indegree(iIP_F, mns_E, 1, -0.8);
//	for(int step = 0; step < step_number; ++step) {
//		connect_fixed_indegree(C_0[step], iIP_F, 1, 0.8);
//	}
//	// reflex arc
//	connect_fixed_indegree(iIP_E, Ia_E, 1, 0.001);
//	connect_fixed_indegree(Ia_aff_E, Ia_E, 1, 0.008);
//	connect_fixed_indegree(mns_E, R_E, 1, 0.00015);
//	connect_fixed_indegree(Ia_E, mns_F, 0.1, -0.002);
////	connect_fixed_indegree(R_E, mns_E, 1, -0.00015);
//	connect_fixed_indegree(R_E, Ia_E, 1, -0.001);
//	//
//	connect_fixed_indegree(iIP_F, Ia_F, 1, 0.001);
//	connect_fixed_indegree(Ia_aff_F, Ia_F, 1, 0.008);
//	connect_fixed_indegree(mns_F, R_F, 1, 0.00015);
////	connect_fixed_indegree(Ia_F, mns_E, 1, -0.08);
//	connect_fixed_indegree(R_F, mns_F, 0.1, -0.00015);
//	connect_fixed_indegree(R_F, Ia_F, 1, -0.001);
//	// todo C_0
//
//	//
//	connect_fixed_indegree(R_E, R_F, 1, -0.04);
//	connect_fixed_indegree(R_F, R_E, 1, -0.04);
//	connect_fixed_indegree(Ia_E, Ia_F, 1, -0.08);
//	connect_fixed_indegree(Ia_F, Ia_E, 1, -0.08);
//	connect_fixed_indegree(iIP_E, iIP_F, 1, -0.04);
//	connect_fixed_indegree(iIP_F, iIP_E, 1, -0.04);

    // TEST
    auto ees = form_group("EES", 1, GENERATOR);
    auto T1 = form_group("T1", 50);
    auto T2 = form_group("T2", 50);
    auto T3 = form_group("T3", 50);
    auto T0 = form_group("T0", 50);


    add_generator(ees, 0, sim_time, 40);
    conn_generator(ees, T0, 1, 0.025);
    connect_fixed_indegree(T0, T1, 1, 0.025);
    connect_fixed_indegree(T1, T2, 1, 0.015);
    connect_fixed_indegree(T2, T1, 1, 0.015);
    connect_fixed_indegree(T1, T3, 4, 0.1);
    connect_fixed_indegree(T2, T3, 2.5, 0.04);
    connect_fixed_indegree(T3, T2, 1, -0.02);
    connect_fixed_indegree(T3, T1, 2, -0.02);


//    connect_fixed_indegree(T0, T1, 1, 0.025);
//    connect_fixed_indegree(T1, T2, 2.5, 10);
//    connect_fixed_indegree(T1, T3, 3.4, 0.00057);
//
//    connect_fixed_indegree(T2, T1, 4, 0.025);
//    connect_fixed_indegree(T2, T3, 115, 999999);
//    connect_fixed_outdegree(T2, fake, 100000, 0);
//
//    connect_fixed_indegree(T3, T1, 20, -0.025);
//    connect_fixed_indegree(T3, T2, 160, -1000);

    save({T3, T1, T2});
   // save(all_groups);
}






int main(int argc, char **argv) {
    custom(init_network, 2);
}