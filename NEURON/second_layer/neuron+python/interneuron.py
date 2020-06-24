from neuron import h
import random
h.load_file('stdlib.hoc') #for h.lambda_f

import random

class interneuron(object):
  '''
  Interneuron class with parameters:
    delay: bool
      Does it have 5ht receptors?
      -Yes: True
      -No: False
    soma: NEURON Section (creates by topol())
    dend: NEURON Section (creates by topol())
    axon: NEURON Section (creates by topol())
    synlistinh: list (creates by synapses())
      list of inhibitory synapses
    synlistex: list (creates by synapses())
      list of excitatory synapses
    synlistees: list (creates by synapses())
      list of excitatory synapses for connection with generators
    x, y, z: int
      3D coordinates (isn't used)
    diffs: list
      list of diffusion mechanisms (NEURON staff)
    recs: list
      list of receptors mechanisms (NEURON staff)
  '''
  def __init__(self, delay):
    self.delay = delay
    self.diffs = []
    self.recs = []
    self.topol()
    self.subsets()
    self.geom()
    self.geom_nseg()
    self.biophys()
    self.synlistinh = []
    self.synlistex = []
    self.synlistees = []
    self.synapses()
    self.x = self.y = self.z = 0.

    def __del__(self):
    #print 'delete ', self
      pass

  def topol(self):
    '''
    Creates sections soma, dend, axon and connects them
    if it's delay creates section dend[]: array
    '''
    self.soma = h.Section(name='soma', cell=self)
    self.axon = h.Section(name='axon', cell= self)
    self.dend = [h.Section(name='dend[%d]' % i) for i in range(random.randint(5,10))]
    for sec in self.dend:
      sec.connect(self.soma(0.5))
    self.axon.connect(self.soma(1))

  def subsets(self):
    '''
    NEURON staff
    adds sections in NEURON SectionList
    '''
    self.all = h.SectionList()
    for sec in h.allsec():
      self.all.append(sec=sec)

  def geom(self):
    '''
    Adds length and diameter to sections
    '''
    self.soma.L = self.soma.diam = random.randint(5, 15) # microns
    self.axon.L = 150 # microns
    self.axon.diam = 1 # microns
    for sec in self.dend:
      sec.L = 200 # microns
      sec.diam = random.gauss(1, 0.1) # microns

  def geom_nseg(self):
    '''
    Calculates numder of segments in section
    '''
    for sec in self.all:
      sec.nseg = int((sec.L/(0.1*h.lambda_f(100)) + .9)/2.)*2 + 1

  def biophys(self):
    '''
    Adds channels and their parameters
    if delay is true, adds 5ht receptors
    '''
    for sec in self.all:
      sec.cm = random.gauss(1, 0.05) # cm uf/cm2 - membrane capacitance

    self.soma.Ra = 100 # Ra ohm cm - membrane resistance
    self.soma.insert('fastchannels')
    self.soma.gnabar_fastchannels = 0.4
    self.soma.gkbar_fastchannels = 0.02
    self.soma.gl_fastchannels = 0.0002
    self.soma.el_fastchannels = -65
    self.soma.insert('extracellular') #adds extracellular mechanism for recording extracellular potential

    for sec in self.dend:
      sec.Ra = 100 # Ra ohm cm - membrane resistance

    for sec in self.dend:
      if self.delay:
        distance = random.uniform(30, 1500)
        sec.insert('hh')
        diff = h.slow_5HT(self.dend(0.5))
        rec = h.r5ht3a(self.dend(0.5))
        rec.gmax = random.uniform(0, 0.002)
        diff.h = random.gauss(distance, distance/5)
        diff.c0cleft = 2
        diff.tx1 = 10+(diff.h/70)*1000
        h.setpointer(diff._ref_serotonin, 'serotonin', rec)
        self.diffs.append(diff)
        self.recs.append(rec)
      else:
        sec.insert('pas')
        sec.g_pas = 0.0002
        sec.e_pas = -65

    self.axon.Ra = 50
    self.axon.insert('hh')

  def position(self, x, y, z):
    '''
    NEURON staff
    Adds 3D position
    '''
    soma.push()
    for i in range(h.n3d()):
      h.pt3dchange(i, x-self.x+h.x3d(i), y-self.y+h.y3d(i), z-self.z+h.z3d(i), h.diam3d(i))
    self.x = x; self.y = y; self.z = z
    h.pop_section()

  def connect2target(self, target):
    '''
    NEURON staff
    Adds presynapses
    Parameters
    ----------
    target: NEURON cell
        target neuron
    Returns
    -------
    nc: NEURON NetCon
        connection between neurons
    '''
    nc = h.NetCon(self.axon(1)._ref_v, target, sec = self.axon)
    nc.threshold = 10
    return nc

  def synapses(self):
    '''
    Adds synapses
    '''
    for sec in self.dend:
      for i in range(50):
        s = h.ExpSyn(self.soma(0.8)) # Excitatory
        s.tau = 0.3
        s.e = 50
        self.synlistex.append(s)
        s = h.Exp2Syn(self.soma(0.5)) # Inhibitory
        s.tau1 = 0.5
        s.tau2 = 3
        s.e = -80
        self.synlistinh.append(s)
        s = h.ExpSyn(sec(0.5)) # Excitatory
        s.tau = 0.3
        s.e = 50
        self.synlistex.append(s)
        s = h.Exp2Syn(sec(0.5)) # Inhibitory
        s.tau1 = 0.5
        s.tau2 = 3
        s.e = -80
        self.synlistinh.append(s)


  def is_art(self):
    return 0
