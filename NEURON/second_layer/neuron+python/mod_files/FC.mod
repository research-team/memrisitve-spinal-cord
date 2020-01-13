TITLE Fast Na K channels
:
: Equations modified from (Traub et al., 1994)

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX fastchannels
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik
	RANGE gnabar, gkbar, vtraub
	RANGE m_inf, h_inf, n_inf
	RANGE tau_m, tau_h, tau_n
	RANGE m_exp, h_exp, n_exp
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gnabar	= 0.3 	(mho/cm2)
	gkbar	= 0.05 	(mho/cm2)

	ena	= 50	(mV)
	ek	= -90	(mV)
	celsius = 36    (degC)
	dt              (ms)
	v               (mV)
}

STATE {
	m h n
}

ASSIGNED {
	ina	(mA/cm2)
	ik	(mA/cm2)
	il	(mA/cm2)
	m_inf
	h_inf
	n_inf
	tau_m
	tau_h
	tau_n
	m_exp
	h_exp
	n_exp
	tadj
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gnabar * m*m*m*h * (v - ena)
	ik  = gkbar * n*n*n*n * (v - ek)
}


DERIVATIVE states {
	evaluate_fct(v)
	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
	n' = (n_inf - n) / tau_n
}

UNITSOFF
INITIAL {
	m = 0
	h = 0
	n = 0
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b

	a = 0.32 * (13-v) / ( exp((13-v)/4) - 1)
	b = 0.28 * (v-40) / ( exp((v-40)/5) - 1)
	tau_m = 1 / (a + b)
	m_inf = a / (a + b)

	a = 0.128 * exp((17-v)/18)
	b = 4 / ( 1 + exp((40-v)/5) )
	tau_h = 1 / (a + b)
	h_inf = a / (a + b)

	a = 0.032 * (15-v) / ( exp((15-v)/5) - 1)
	b = 0.5 * exp((10-v)/40)
	tau_n = 1 / (a + b)
	n_inf = a / (a + b)

	m_exp = 1 - exp(-dt/tau_m)
	h_exp = 1 - exp(-dt/tau_h)
	n_exp = 1 - exp(-dt/tau_n)
}

UNITSON
