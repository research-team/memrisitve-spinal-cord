"""
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011).
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press.
DOI:10.1017/CBO9780511975899

Based on the NEURON repository
"""

import numpy as np
import matplotlib.pyplot as plt
import logging as log

log.basicConfig(level=log.INFO)

test_inter = False
debug = False
dt = 0.025  # [ms] - sim step
nrns_number = 1
nrns = list(range(nrns_number))

sim_time = 50
stimulus = (np.arange(11, sim_time, 25) / dt).astype(int)

Re = 333        # [Ohm cm] Resistance of extracellular space
E_Na = 50       # [mV] - reversal potential
E_K = -80       # [mV] - reversal potential
E_L = -70       # [mV] - reversal potential
E_ex = 50       # [mV] - reversal potential
E_in = -100     # [mV] reverse inh
V_adj = -63     # [mV] - adjust voltage for -55 threshold
tau_syn_exc = 0.3       # [ms]
tau_syn_inh = 2.0       # [ms]
ref_time = int(3 / dt)  # [steps]

if test_inter:
	Cm = 1          # [uF/cm2] membrane capacity
	g_Na = 120      # [S / cm2]
	g_K = 36        # [S / cm2]
	g_L = 0.3       # [S / cm2]
	Ra = 100        # [Ohm cm]
	diam = 6        # [um]
	dx = 2          # [um]
else:
	Cm = 2          # [uF/cm2] membrane capacity
	gnabar = 0.05   # [S/cm2]
	g_L = 0.002     # [S/cm2]
	gkrect = 0.3    # [S/cm2]
	gcaN = 0.05     # const ???
	gcaL = 0.0001   # const ???
	gcak = 0.3      # const ???
	Ra = 200        # [Ohm cm]
	diam = 50       # [um] diameter
	dx = 50         # [um] compartment length
	ca0 = 2         # const ???
	amA = 0.4       # const ???
	amB = 66        # const ???
	amC = 5         # const ???
	bmA = 0.4       # const ???
	bmB = 32        # const ???
	bmC = 5         # const ???
	R = 8.314472    # (k-mole) (joule/degC) const
	F = 96485.34    # (faraday) (kilocoulombs) const

_nt_ncell = 1       # analogous to old rootnodecount
nnode = 2           # Number of nodes for ith section
_nt_end = 3         # 1 + position of last in v_node array
segs = list(range(_nt_end))

i1 = 0
i2 = i1 + _nt_ncell
i3 = _nt_end

nrn_shape = (nrns_number, _nt_end)

Vm = np.full(nrn_shape, -70, dtype=np.float)        # [mV] - array for three compartments volatge
n = np.full(nrn_shape, 0, dtype=np.float)       # [0..1] compartments channel
m = np.full(nrn_shape, 0, dtype=np.float)       # [0..1] compartments channel
h = np.full(nrn_shape, 0, dtype=np.float)        # [0..1] compartments channel
cai = np.full(nrn_shape, 0, dtype=np.float)    # [0..1] compartments channel
hc = np.full(nrn_shape, 1, dtype=np.float)      # [0..1] compartments channel
mc = np.full(nrn_shape, 0, dtype=np.float)     # [0..1] compartments channel
p = np.full(nrn_shape, 0, dtype=np.float)     # [0..1] compartments channel

I_K = np.full(nrn_shape, 0, dtype=np.float)         # [nA] ionic currents
I_Na = np.full(nrn_shape, 0, dtype=np.float)        # [nA] ionic currents
I_L = np.full(nrn_shape, 0, dtype=np.float)         # [nA] ionic currents
I_Ca = np.full(nrn_shape, -0.0004, dtype=np.float)  # [nA] ionic currents
g_exc = np.full(nrns_number, 0, dtype=np.float)     # [S] conductivity level
g_inh = np.full(nrns_number, 0, dtype=np.float)     # [S] conductivity level
ref_time_timer = np.full(nrns_number, 0, dtype=np.float)   # [steps] refractory period timer

E_Ca = np.full(nrn_shape, 131, dtype=np.float)      # [mV]
old_Vm = np.full(nrns_number, -70, dtype=np.float)  # [mV] old value of Vm

NODE_RHS = np.zeros(shape=nrn_shape, dtype=np.float) # right hand side in node equation
NODE_D = np.zeros(shape=nrn_shape, dtype=np.float)   # diagonal element in node equation
NODE_A = np.zeros(shape=nrn_shape, dtype=np.float)   # is the effect of this node on the parent node's equation
NODE_B = np.zeros(shape=nrn_shape, dtype=np.float)   # is the effect of the parent node on this node's equation
NODE_RINV = np.zeros(shape=nrn_shape, dtype=np.float)   # is the effect of the parent node on this node's equation
NODE_AREA = np.zeros(shape=nrn_shape, dtype=np.float)   # is the effect of the parent node on this node's equation

spikes = []
# todo recheck
const3 = (np.log(np.sqrt(dx ** 2 + diam ** 2) + dx) - np.log(np.sqrt(dx ** 2 + diam ** 2) - dx)) / (4 * np.pi * dx * Re)

axes_names = 'cai il ina ik ica Eca m h n p mc hc A0 B0 D0 RINV0 Vm0 RHS0 A1 B1 D1 RINV1 Vm1 RHS1 A2 B2 D2 RINV2 Vm2 RHS2'
debug_headers = "iter Vm vv1 vv2 INa IK1 IK2 IL ECa ICa1 ICa2 m h p n mc hc cai".split()
strformat = "{:<15.6f}" * len(debug_headers)
headformat = "{:<15}" * len(debug_headers)
rows = 5
cols = 6
GRAS_data = []
sim_time_steps = int(sim_time / dt)  # simulation time

def get_neuron_data():
	with open("/home/alex/NEURTEST/tablelog") as file:
		file.readline()
		neuron_data = np.array([line.split("\t") for line in file]).astype(float)
	return neuron_data

########################
data = get_neuron_data()

def Exp(volt):
	if volt < -100:
		return 0
	return np.exp(volt)

def alpham(volt):
	if abs((volt + amB) / amC) < 1e-6:
		return amA * amC
	return (amA * (volt + amB)) / (1 - Exp(-(volt + amB) / amC))

def betam(volt):
	if abs((volt + bmB) / bmC) < 1e-6:
		return -bmA * bmC
	return (bmA * (-(volt + bmB))) / (1 - Exp((volt + bmB) / bmC))

def syn_current(nrn, voltage):
	"""

	"""
	return g_exc[nrn] * (voltage - E_ex) + g_inh[nrn] * (voltage - E_in)

def nrn_current(nrn, seg, voltage):
	"""

	"""
	ina = gnabar * m[nrn, seg] ** 3 * h[nrn, seg] * (voltage - E_Na)
	ik = gkrect * n[nrn, seg] ** 4 * (voltage - E_K) + gcak * cai[nrn, seg] ** 2 / (cai[nrn, seg] ** 2 + 0.014 ** 2) * (voltage - E_K)
	il = g_L * (voltage - E_L)
	Eca = (1000 * R * 309.15 / (2 * F)) * np.log(ca0 / cai[nrn, seg])
	ica = gcaN * mc[nrn, seg] ** 2 * hc[nrn, seg] * (voltage - Eca) + gcaL * p[nrn, seg] * (voltage - Eca)
	# save
	I_Na[nrn, seg] = ina
	I_K[nrn, seg] = ik
	I_L[nrn, seg] = il
	E_Ca[nrn, seg] = Eca
	I_Ca[nrn, seg] = ica

	return ina + ik + il + ica

def recalc_synaptic(nrn):
	"""

	"""
	g_exc[nrn] -= g_exc[nrn] / tau_syn_exc * dt
	g_inh[nrn] -= g_inh[nrn] / tau_syn_inh * dt

def nrn_initial(nrn, seg, V):
	"""
	evaluate_fct cropped
	"""
	a = alpham(V)
	b = betam(V)
	m_inf = a / (a + b)
	h_inf = 1 / (1 + Exp((V + 65) / 7))
	n_inf = 1 / (1 + Exp(-(V + 38) / 15))
	mc_inf = 1 / (1 + Exp(-(V + 32) / 5))
	hc_inf = 1 / (1 + Exp((V + 50) / 5))
	p_inf = 1 / (1 + Exp(-(V + 55.8) / 3.7))
	"""INITIAL"""
	m[nrn, seg] = m_inf
	h[nrn, seg] = h_inf
	p[nrn, seg] = p_inf
	n[nrn, seg] = n_inf
	mc[nrn, seg] = mc_inf
	hc[nrn, seg] = hc_inf
	cai[nrn, seg] = 0.0001

def recalc_channels(nrn, seg, V):
	"""

	"""
	# BREAKPOINT -> states -> evaluate_fct
	"""evaluate_fct"""
	a = alpham(V)
	b = betam(V)
	# m
	tau_m = 1 / (a + b)
	m_inf = a / (a + b)
	# h
	tau_h = 30 / (Exp((V + 60) / 15) + Exp(-(V + 60) / 16))
	h_inf = 1 / (1 + Exp((V + 65) / 7))
	# DELAYED RECTIFIER POTASSIUM
	tau_n = 5 / (Exp((V + 50) / 40) + Exp(-(V + 50) / 50))
	n_inf = 1 / (1 + Exp(-(V + 38) / 15))
	# CALCIUM DYNAMICS N-type
	tau_mc = 15
	mc_inf = 1 / (1 + Exp(-(V + 32) / 5))
	tau_hc = 50
	hc_inf = 1 / (1 + Exp((V + 50) / 5))
	# CALCIUM DYNAMICS L-type
	tau_p = 400
	p_inf = 1 / (1 + Exp(-(V + 55.8) / 3.7))
	"""states"""
	m[nrn, seg] += (1 - np.exp(-dt * (1 / tau_m))) * (m_inf / tau_m / (1 / tau_m) - m[nrn, seg])
	h[nrn, seg] += (1 - np.exp(-dt * (1 / tau_h))) * (h_inf / tau_h / (1 / tau_h) - h[nrn, seg])
	p[nrn, seg] += (1 - np.exp(-dt * (1 / tau_p))) * (p_inf / tau_p / (1 / tau_p) - p[nrn, seg])
	n[nrn, seg] += (1 - np.exp(-dt * (1 / tau_n))) * (n_inf / tau_n / (1 / tau_n) - n[nrn, seg])
	mc[nrn, seg] += (1 - np.exp(-dt * (1 / tau_mc))) * (mc_inf / tau_mc / (1 / tau_mc) - mc[nrn, seg])
	hc[nrn, seg] += (1 - np.exp(-dt * (1 / tau_hc))) * (hc_inf / tau_hc / (1 / tau_hc) - hc[nrn, seg])
	cai[nrn, seg] += (1 - np.exp(-dt * 0.04)) * (-0.01 * I_Ca[nrn, seg] / 0.04 - cai[nrn, seg])

	assert 0 <= cai[nrn, seg] <= 0.1
	assert -200 <= Vm[nrn, seg] <= 200
	assert 0 <= m[nrn, seg] <= 1
	assert 0 <= n[nrn, seg] <= 1
	assert 0 <= h[nrn, seg] <= 1
	assert 0 <= p[nrn, seg] <= 1
	assert 0 <= mc[nrn, seg] <= 1
	assert 0 <= hc[nrn, seg] <= 1

def save_data(to_end=False):
	if to_end:
		GRAS_data[-1] += [NODE_A[0, 0], NODE_B[0, 0], NODE_D[0, 0], NODE_RINV[0, 0], Vm[0, 0], NODE_RHS[0, 0],
		                  NODE_A[0, 1], NODE_B[0, 1], NODE_D[0, 1], NODE_RINV[0, 1], Vm[0, 1], NODE_RHS[0, 1],
		                  NODE_A[0, 2], NODE_B[0, 2], NODE_D[0, 2], NODE_RINV[0, 2], Vm[0, 2], NODE_RHS[0, 2]]
	else:
		MID = 1
		GRAS_data.append([cai[0, MID], I_L[0, MID], I_Na[0, MID], I_K[0, MID], I_Ca[0, MID],
		                  E_Ca[0, MID], m[0, MID], h[0, MID], n[0, MID], p[0, MID], mc[0, MID], hc[0, MID]])

def nrn_rhs(nrn, t):
	"""
	void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	calculate right hand side of
	cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	This is a common operation for fixed step, cvode, and daspk methods
	"""
	# init _rhs and _lhs (NODE_D) as zero
	for nd in range(i1, i3):
		NODE_RHS[nrn, nd] = 0
		NODE_D[nrn, nd] = 0
	# update MOD rhs, CAPS has no current [CAP MOD CAP]!
	for seg in segs:
		if seg == 0 or seg == nnode:
			continue
		if t == -1:
			V = -70
		else:
			V = data[t, 22] # Vm[nrn, seg]
# SYNAPTIC update
		# static void nrn_cur
		_g = syn_current(nrn, V + 0.001)
		_rhs = syn_current(nrn, V)
		_g = (_g - _rhs) / .001
		_g *= 1.e2 / NODE_AREA[nrn, seg]
		_rhs *= 1.e2 / NODE_AREA[nrn, seg]
		NODE_RHS[nrn, seg] -= _rhs
		# todo check info about _g updating (where is it stored?)
		# static void nrn_jacob
		NODE_D[nrn, seg] += _g

		# NEURON update
		# static void nrn_cur
		_g = nrn_current(nrn, seg, V + 0.001)
		_rhs = nrn_current(nrn, seg, V)
		# save data like in NEURON (after .mod nrn_cur)
		save_data()
		_g = (_g - _rhs) / 0.001
		NODE_RHS[nrn, seg] -= _rhs
		# todo check info about _g updating (where is it stored?)
		# static void nrn_jacob
		NODE_D[nrn, seg] += _g
	# end FOR segments

	# activsynapse_rhs()
	# NODE_rhs[nrn, seg] += 0
	# if EXTRA: nrn_rhs_ext(_nt);
	# NODE_rhs[nrn, seg] += 0
	# activstim_rhs()
	# NODE_rhs[nrn, seg] += 0
	# activclamp_rhs()
	# NODE_rhs[nrn, seg] += 0

	# todo always 0, because Vm0 = Vm1 = Vm2 at CAP node CAP model
	for nd in range(i2, i3):
		pnd = nd - 1
		# double dv = NODEV(pnd) - NODEV(nd);
		dv = Vm[nrn, pnd] - Vm[nrn, nd]
		# fixme remove after Vm fixing
		dv = 0
		# our connection coefficients are negative so
		NODE_RHS[nrn, nd] -= NODE_B[nrn, nd] * dv
		NODE_RHS[nrn, pnd] += NODE_A[nrn, nd] * dv

def nrn_lhs(nrn):
	"""
	void nrn_lhs(NrnThread *_nt)
	NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	"""
	# nt->cj = 2/dt if (secondorder) else 1/dt
	# note, the first is CAP
	# nrn_cap_jacob(_nt, _nt->tml->ml);
	cj = 1 / dt
	cfac = 0.001 * cj
	nodecount = 1
	# fixme +1 for nodelist
	for nd in range(0 + 1, nodecount + 1):
		NODE_D[nrn, nd] += cfac * Cm

	# activsynapse_lhs()
	# NODE_D[nrn, seg] += 0
	# nrn_setup_ext(_nt);
	# NODE_D[nrn, seg] += 0
	# activclamp_lhs();
	# NODE_D[nrn, seg] += 0

	# updating D
	for nd in range(i2, i3):
		pnd = nd - 1
		NODE_D[nrn, nd] -= NODE_B[nrn, nd]
		NODE_D[nrn, pnd] -= NODE_A[nrn, nd]

def bksub(nrn):
	"""
	void bksub(NrnThread* _nt)
	"""
	for nd in range(i1, i2):
		NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]
	for nd in range(i2, i3):
		pnd = nd - 1
		NODE_RHS[nrn, nd] -= NODE_B[nrn, nd] * NODE_RHS[nrn, pnd]
		NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]

def triang(nrn):
	"""
	void triang(NrnThread* _nt)
	"""
	nd = i3 - 1
	while nd >= i2:
		pnd = nd - 1
		ppp = NODE_A[nrn, nd] / NODE_D[nrn, nd]
		NODE_D[nrn, pnd] -= ppp * NODE_B[nrn, nd]
		NODE_RHS[nrn, pnd] -= ppp * NODE_RHS[nrn, nd]
		nd -= 1

def nrn_solve(nrn):
	"""
	void nrn_solve(NrnThread* _nt)
	"""
	triang(nrn)
	bksub(nrn)

def setup_tree_matrix(nrn, t):
	"""
	void setup_tree_matrix(NrnThread* _nt)
	"""
	nrn_rhs(nrn, t)
	nrn_lhs(nrn)

def update(nrn):
	"""
	void update(NrnThread* _nt)
	"""
	for nd in range(i1, _nt_end):
		Vm[nrn, nd] += NODE_RHS[nrn, nd]

	# save data like in NEURON (after .mod nrn_cur)
	save_data(to_end=True)
	# check on spike
	# if ref_time_timer[nrn_id] == 0 and -55 <= Vm[nrn_id, 0]:
	# 	ref_time_timer[nrn_id] = ref_time
	# spikes.append(t * dt)
	# update refractory period timer
	if ref_time_timer[nrn] > 0:
		ref_time_timer[nrn] -= 1

def nrn_fixed_step_lastpart(nrn, t):
	# new V
	V = data[t + 1][22]
	recalc_synaptic(nrn)
	for seg in segs:
		recalc_channels(nrn, seg, V)

def NET_RECEIVE(t):
	if t in stimulus:
		g_exc[0] += 5.5  # [uS]

def nrn_area_ri():
	"""
	void nrn_area_ri(Section *sec) [790] treeset.c
	area for right circular cylinders. Ri as right half of parent + left half of this
	"""
	for nrn in nrns:
		rright = 0
		# todo sec->pnode needs +1 index
		for nd in range(0 + 1, nnode - 1 + 1):
			NODE_AREA[nrn, nd] = np.pi * dx * diam
			rleft = 1e-2 * Ra * (dx / 2) / (np.pi * diam * diam / 4) # left half segment Megohms
			NODE_RINV[nrn, nd] = 1 / (rleft + rright) # uS
			rright = rleft
		nd = nnode - 1 + 1
		# last segment has 0 length. area is 1e2 in dimensionless units
		# NODE_AREA[nrn, nd] = 1.e2
		NODE_AREA[:, 0] = 100
		NODE_AREA[:, nd] = 100
		NODE_RINV[nrn, nd] = 1 / rright

def connection_coef():
	"""
	void connection_coef(void)  [854] treeset.c
	"""
	nrn_area_ri()
	# set NODE_A and NODE_B
	# NODE_A is the effect of this node on the parent node's equation
	# NODE_B is the effect of the parent node on this node's equation
	for nrn in nrns:
		# first the effect of node on parent equation. Note That
		# last nodes have area = 1.e2 in dimensionless units so that
		# last nodes have units of microsiemens
		#todo sec->pnode needs +1 index
		nd = 1
		area = NODE_AREA[nrn, nd - 1] # parentnode
		# ClassicalNODEA
		# sec->prop->dparam[4].val = 1
		NODE_A[nrn, nd] = -1.e2 * 1 * NODE_RINV[nrn, nd] / area
		# todo sec->pnode needs +1 index
		for j in range(1 + 1, nnode + 1):
			nd = j
			pnd = j - 1
			# ClassicalNODEA
			NODE_A[nrn, nd] = -1.e2 * NODE_RINV[nrn, nd] / NODE_AREA[nrn, pnd]
		# now the effect of parent on node equation
		# todo sec->pnode needs +1 index
		for nd in range(0 + 1, nnode + 1):
			NODE_B[nrn, nd] = -1.e2 * NODE_RINV[nrn, nd] / NODE_AREA[nrn, nd]

def finitialize(v_init=-70):
	"""

	"""
	connection_coef()
	for nrn in nrns:
		for seg in segs:
			nrn_initial(nrn, seg, v_init)
		setup_tree_matrix(nrn, -1)
	# clean saved data
	GRAS_data.clear()
	print(NODE_A[0, :])
	print(NODE_RHS[0, :])

def nrn_fixed_step_thread(t):
	"""
	void *nrn_fixed_step_thread(NrnThread *nth)
	"""
	print(f"=========== {t} ==============")

	# update data for each neuron
	for nrn in nrns:
		# add stimulus
		NET_RECEIVE(t)
		setup_tree_matrix(nrn, t)
		nrn_solve(nrn)
		update(nrn)
		nrn_fixed_step_lastpart(nrn, t)

	print("=========== END UPDATE ==============")

def simulation():
	"""

	"""
	finitialize()
	# start simulation loop
	for t in range(sim_time_steps - 2):
		nrn_fixed_step_thread(t)

def plot(gras_data, neuron_data):
	"""

	"""
	plt.close()
	fig, ax = plt.subplots(rows, cols, sharex='all')
	xticks = np.arange(neuron_data.shape[0]) * dt
	# plot NEURON and GRAS
	for index, (neuron_d, gras_d, name) in enumerate(zip(neuron_data.T, gras_data.T, axes_names.split())):
		row = int(index / cols)
		col = index % cols
		if row >= rows:
			break
		if all(np.abs(neuron_d - gras_d) <= 5 * 1e-5):
			ax[row, col].plot(xticks, neuron_d, label='NEURON', lw=3, ls='--')
		else:
			print(name, max(np.abs(neuron_d - gras_d)))
			ax[row, col].plot(xticks, neuron_d, label='NEURON', lw=3)
			ax[row, col].plot(xticks, gras_d, label='GRAS', lw=1, color='r')
		# ax[row, col].plot(spikes, [np.mean(neuron_data)] * len(spikes), '.', color='r', ms=10)
		# ax[row, col].plot(spikes, [np.mean(gras_data)] * len(spikes), '.', color='b', ms=5)
		passed = np.abs(np.mean(neuron_d) / np.mean(gras_d)) * 100 - 100
		ax[row, col].set_title(f"{name} max diff {passed:.3f}")
		ax[row, col].set_xlim(0, sim_time)
	plt.show()


if __name__ == "__main__":
	simulation()
	GRAS_data = np.array(GRAS_data)
	xlength = GRAS_data.shape[0]
	NEURON_data = get_neuron_data()[:xlength, :]
	log.info(f"GRAS shape {GRAS_data.shape}")
	log.info(f"NEURON shape {NEURON_data.shape}")
	plot(GRAS_data, NEURON_data)
