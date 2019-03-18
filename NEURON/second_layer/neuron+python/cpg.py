import logging
logging.basicConfig(level=logging.DEBUG)
from neuron import h
h.load_file('nrngui.hoc')

#paralleling
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

#param
speed = 25
EES_i = 25
version = 1 

moto_EX_v_vec = []
moto_FL_v_vec = []
soma_v_vec = []


from interneuron import interneuron
from motoneuron import motoneuron
from bioaff import bioaff

import random


# network creation

class cpg:
  def __init__(self, speed, EES_int, inh_p, N = 20):

    self.interneurons = []
    self.motoneurons = []
    self.afferents = []
    self.stims = []
    self.ncell = N

    nMN = 168
    nAff = 120
    nInt = 196
    
    D1_1E = self.addpool(self.ncell, True)
    D1_2E = self.addpool(self.ncell, True)
    D1_3E = self.addpool(self.ncell, True)
    D1_4E = self.addpool(self.ncell, True)
    D1_1F = self.addpool(self.ncell, True)
    D1_2F = self.addpool(self.ncell, True)
    D1_3F = self.addpool(self.ncell, True)
    D1_4F = self.addpool(self.ncell, True)
  
    D2_1E = self.addpool(self.ncell, True)
    D2_2E = self.addpool(self.ncell, True)
    D2_3E = self.addpool(self.ncell, True)
    D2_4E = self.addpool(self.ncell, True)
    D2_1F = self.addpool(self.ncell, True)
    D2_2F = self.addpool(self.ncell, True)
    D2_3F = self.addpool(self.ncell, True)
    D2_4F = self.addpool(self.ncell, True)

    D3_1 = self.addpool(self.ncell, True)
    D3_2 = self.addpool(self.ncell, True)
    D3_3 = self.addpool(self.ncell, True)
    D3_4 = self.addpool(self.ncell, True)

    D4_1E = self.addpool(self.ncell, True)
    D4_2E = self.addpool(self.ncell, True)
    D4_3E = self.addpool(self.ncell, True)
    D4_4E = self.addpool(self.ncell, True)
    D4_1F = self.addpool(self.ncell, True)
    D4_2F = self.addpool(self.ncell, True)
    D4_3F = self.addpool(self.ncell, True)
    D4_4F = self.addpool(self.ncell, True)

    D5_1 = self.addpool(self.ncell, True)
    D5_2 = self.addpool(self.ncell, True)
    D5_3 = self.addpool(self.ncell, True)
    D5_4 = self.addpool(self.ncell, True)

    G1_1 = self.addpool(self.ncell, False)
    G1_2 = self.addpool(self.ncell, False)
    G1_3 = self.addpool(self.ncell, False)

    G2_1 = self.addpool(self.ncell, False)
    G2_2 = self.addpool(self.ncell, False)
    G2_3 = self.addpool(self.ncell, False)
   
    G3_1E = self.addpool(self.ncell, False)
    G3_2E = self.addpool(self.ncell, False)
    G3_3E = self.addpool(self.ncell, False)
    G3_1F = self.addpool(self.ncell, False)
    G3_2F = self.addpool(self.ncell, False)
    G3_3F = self.addpool(self.ncell, False)

    G4_1 = self.addpool(self.ncell, False)
    G4_2 = self.addpool(self.ncell, False)
    G4_3 = self.addpool(self.ncell, False)

    G5_1 = self.addpool(self.ncell, False)
    G5_2 = self.addpool(self.ncell, False)
    G5_3 = self.addpool(self.ncell, False)
   
    E1 = self.addpool(self.ncell, False)
    E2 = self.addpool(self.ncell, False)
    E3 = self.addpool(self.ncell, False)
    E4 = self.addpool(self.ncell, False)

    I3_E = self.addpool(self.ncell, False)
    I4_E = self.addpool(self.ncell, False)
    I5_E = self.addpool(self.ncell, False)
    I5_F = self.addpool(self.ncell, False)
        
    sens_aff = self.addafferents(nAff)
    Ia_aff_E = self.addafferents(nAff)
    Ia_aff_F = self.addafferents(nAff)

    self.mns_E = self.addmotoneurons(nMN)
    self.mns_F = self.addmotoneurons(nMN)

    #interneuronal pool
    IP1_E = self.addpool(self.ncell, True)
    IP1_F = self.addpool(self.ncell, True)
    IP2_E = self.addpool(self.ncell, True)
    IP2_F = self.addpool(self.ncell, True)
    IP3_E = self.addpool(self.ncell, True)
    IP3_F = self.addpool(self.ncell, True)
    IP4_E = self.addpool(self.ncell, True)
    IP4_F = self.addpool(self.ncell, True)
    IP5_E = self.addpool(self.ncell, True)
    IP5_F = self.addpool(self.ncell, True)

    # EES
    self.ees = self.addgener(1, EES_int, 10000)
    self.dees = self.addgener(speed, EES_int, (speed/25)*4)
    self.tees = self.addgener(2*speed, EES_int, (speed/25)*2)

    
    #skin inputs
    c_int = 5

    C1 = self.addgener(speed*0, c_int, speed/c_int)
    C2 = self.addgener(speed*1, c_int, speed/c_int)
    C3 = self.addgener(speed*2, c_int, speed/c_int)
    C4 = self.addgener(speed*3, c_int, 2*speed/c_int)
    C5 = self.addgener(speed*5, c_int, speed/c_int)

    C_1 = self.addgener(speed*0, c_int, 6*speed/c_int)
    C_0 = self.addgener(speed*6, c_int, 125/c_int)

    #reflex arc
    Ib_E = self.addpool(nInt, False)
    Ia_E = self.addpool(nInt, False)
    R_E = self.addpool(nInt, False)

    Ib_F = self.addpool(nInt, False)
    Ia_F = self.addpool(nInt, False)
    R_F = self.addpool(nInt, False)
  
    #delays
    connectdelay_extensor(D1_1E, D1_2E, D1_3E, D1_4E)
    connectdelay_flexor(D1_1F, D1_2F, D1_3F, D1_4F)

    connectdelay_extensor(D2_1E, D2_2E, D2_3E, D2_4E)
    connectdelay_flexor(D2_1F, D2_2F, D2_3F, D2_4F)

    connectdelay_extensor(D3_1, D3_2, D3_3, D3_4)

    connectdelay_extensor(D4_1E, D4_2E, D4_3E, D4_4E)
    connectdelay_flexor(D4_1F, D4_2F, D4_3F, D4_4F)

    connectdelay_extensor(D5_1, D5_2, D5_3, D5_4)

    #generators
    connectgenerator(G1_1, G1_2, G1_3)
    
    connectgenerator(G2_1, G2_2, G2_3)

    connectgenerator(G3_1E, G3_2E, G3_3E)
    connectgenerator(G3_1F, G3_2F, G3_3F)

    connectgenerator(G4_1, G4_2, G4_3)

    connectgenerator(G5_1, G5_2, G5_3)

    #between delays (FLEXOR)
    exconnectcells(D2_3F, D3_1, 0.00015, 1, 27)
    exconnectcells(D2_3F, D3_4, 0.00025, 1, 27)
    exconnectcells(D4_3F, D5_1, 0.0002, 1, 27)
    exconnectcells(D4_3F, D5_4, 0.00025, 1, 27)

    #between delays via excitatory pools
    #extensor
    exconnectcells(D1_3E, E1, 0.5, 1, 27)
    exconnectcells(E1, E2, 0.5, 1, 27)
    exconnectcells(E2, E3, 0.5, 1, 27)
    exconnectcells(E3, E4, 0.5, 1, 27)

    connectexpools_extensor(D2_1E, D2_4E, E1)
    connectexpools_extensor(D3_1, D3_4, E2)
    connectexpools_extensor(D4_1E, D4_4E, E3)
    connectexpools_extensor(D5_1, D5_4, E4)

    #flexor
    exconnectcells(D1_3F, E1, 0.5, 1, 27)
    exconnectcells(E1, E2, 0.5, 1, 27)
    exconnectcells(E2, E3, 0.5, 1, 27)
    exconnectcells(E3, E4, 0.5, 1, 27)

    connectexpools_extensor(D2_1F, D2_4F, E1)
    connectexpools_extensor(D3_1, D3_4, E2)
    connectexpools_extensor(D4_1F, D4_4F, E3)
    connectexpools_extensor(D5_1, D5_4, E4)

    #delay -> generator
    #extensor
    exconnectcells(D1_3E, G1_1, 0.05, 2, 27)
    exconnectcells(D2_3E, G2_1, 0.05, 2, 27)
    exconnectcells(D3_3, G3_1E, 0.05, 2, 27)
    exconnectcells(D4_3E, G4_1, 0.05, 2, 27)
    exconnectcells(D5_3, G5_1, 0.05, 1, 27)

    #flexor
    exconnectcells(D1_3F, G1_1, 0.05, 2, 27)
    exconnectcells(D1_3F, G2_1, 0.05, 2, 27)
    exconnectcells(D2_3F, G2_1, 0.05, 2, 27)
    exconnectcells(D3_3, G3_1F, 0.05, 2, 27)
    exconnectcells(D4_3F, G4_1, 0.05, 2, 27)
    exconnectcells(D5_3, G5_1, 0.05, 2, 27)

    #between generators (FLEXOR)
    exconnectcells(G3_1F, G4_1, 0.05, 2, 27)
    exconnectcells(G3_2F, G4_1, 0.05, 2, 27)

    #inhibitory projections
    #extensor
    inhconnectcells(I3_E, G1_2, 0.8, 1, 27)
    inhconnectcells(I3_E, G1_1, 0.8, 1, 27)

    inhconnectcells(I4_E, G2_1, 0.8, 1, 27)
    inhconnectcells(I4_E, G2_2, 0.8, 1, 27)
    inhconnectcells(I4_E, G1_2, 0.8, 1, 27)
    inhconnectcells(I4_E, G1_1, 0.8, 1, 27)
    inhconnectcells(I4_E, G3_1E, 0.8, 1, 27)
    inhconnectcells(I4_E, G3_2E, 0.8, 1, 27)

    inhconnectcells(I5_E, G1_1, 0.8, 1, 27)
    inhconnectcells(I5_E, G1_2, 0.8, 1, 27)
    inhconnectcells(I5_E, G2_1, 0.8, 1, 27)
    inhconnectcells(I5_E, G2_2, 0.8, 1, 27)
    inhconnectcells(I5_E, G3_1E, 0.8, 1, 27)
    inhconnectcells(I5_E, G3_2E, 0.8, 1, 27)
    inhconnectcells(I5_E, G4_1, 0.8, 1, 27)
    inhconnectcells(I5_E, G4_2, 0.8, 1, 27)
    
    #EES
    genconnect(sens_aff, self.ees, 1, 0, 50)
    genconnect(Ia_aff_F, self.ees, 1, 0, 50)

    exconnectcells(sens_aff, D1_1E, 0.5, 0, 50)
    exconnectcells(sens_aff, D1_4E, 0.5, 0, 50)

    exconnectcells(sens_aff, D1_1F, 0.5, 0, 50)
    exconnectcells(sens_aff, D1_4F, 0.5, 0, 50)

    genconnect(Ia_aff_E[:int(len(Ia_aff_E)/3)], self.ees, 0.2, 1, 60)
    genconnect(Ia_aff_E[:int(len(Ia_aff_E)/2)], self.dees, 0.5, 2, 50)
    genconnect(Ia_aff_E, self.tees, 1.5, 1, 50)

    exconnectcells(Ia_aff_E[:int(len(Ia_aff_E)/3)], self.mns_E[:int(len(self.mns_E)/3 + random.randint(5, 10))], 0.5, 1, 50)
    exconnectcells(Ia_aff_E[:int(len(Ia_aff_E)/2)], self.mns_E[:int(len(self.mns_E)/2 + random.randint(5, 10))], 0.5, 1, 50)
    exconnectcells(Ia_aff_E, self.mns_E, 0.8, 1, 50)

    exconnectcells(Ia_aff_F, self.mns_F, 0.5, 1, 50)

    #IP
    #Extensor
    exconnectcells(G1_1, IP1_E, 0.5, 2, 50)
    exconnectcells(G1_2, IP1_E, 0.5, 2, 50)

    exconnectcells(G2_1, IP2_E, 0.5, 2, 50)
    exconnectcells(G2_2, IP2_E, 0.5, 2, 50)

    exconnectcells(G3_1E, IP3_E, 0.5, 2, 50)
    exconnectcells(G3_2E, IP3_E, 0.5, 2, 50)

    exconnectcells(G4_1, IP4_E, 0.5, 2, 50)
    exconnectcells(G4_2, IP4_E, 0.5, 2, 50)

    exconnectcells(G5_1, IP5_E, 0.5, 1, 50)
    exconnectcells(G5_2, IP5_E, 0.5, 1, 50)
    

    exconnectcells(IP1_E, self.mns_E[:int(len(self.mns_E)/5 - random.randint(0, 5))], 0.5, 2, 60)
    exconnectcells(IP2_E, self.mns_E[int(len(self.mns_E)/5):int(2*len(self.mns_E)/5 + random.randint(5, 10))], 0.7, 2, 60)
    exconnectcells(IP3_E, self.mns_E, 0.8, 2, 80)
    exconnectcells(IP4_E, self.mns_E[int(2*len(self.mns_E)/5):], 0.8, 2, 80)
    exconnectcells(IP5_E, self.mns_E[int(3*len(self.mns_E)/5):], 0.75, 1, 80)

    #Flexor
    exconnectcells(G1_1, IP1_F, 0.5, 2, 50)
    exconnectcells(G1_2, IP1_F, 0.5, 2, 50)

    exconnectcells(G2_1, IP2_F, 0.5, 2, 50)
    exconnectcells(G2_2, IP2_F, 0.5, 2, 50)

    exconnectcells(G3_1F, IP3_F, 0.5, 2, 50)
    exconnectcells(G3_2F, IP3_F, 0.5, 2, 50)

    exconnectcells(G4_1, IP4_F, 0.5, 2, 50)
    exconnectcells(G4_2, IP4_F, 0.5, 2, 50)

    exconnectcells(G5_1, IP5_F, 0.5, 2, 50)
    exconnectcells(G5_2, IP5_F, 0.5, 2, 50)
    
    exconnectcells(IP1_F, self.mns_F[:2*int(len(self.mns_F)/5)], 0.8, 2, 60)
    exconnectcells(IP2_F, self.mns_F[int(len(self.mns_F)/5):int(3*len(self.mns_E)/5 + random.randint(5, 10))], 0.8, 2, 60)
    exconnectcells(IP3_F, self.mns_F, 0.8, 2, 80)
    exconnectcells(IP4_F, self.mns_F[int(2*len(self.mns_F)/5):], 0.8, 2, 80)
    exconnectcells(IP5_F, self.mns_F[int(3*len(self.mns_F)/5):], 0.8, 1, 80)

    #skin inputs
    #C1
    genconnect(D1_1E, C1, 0.0001, 1, 50)
    genconnect(D1_4E, C1, 0.0001, 1, 50)

    #C2
    genconnect(D1_1E, C2, 0.00035, 1, 27)
    genconnect(D1_4E, C2, 0.00035, 1, 27)
    genconnect(D2_1E, C2, 0.00045, 1, 27)
    genconnect(D2_4E, C2, 0.00045, 1, 27)

    #C3
    genconnect(D2_1E, C3, 0.00048, 1, 27)
    genconnect(D2_4E, C3, 0.00048, 1, 27)
    genconnect(D3_1, C3, 0.00048, 1, 27)
    genconnect(D3_2, C3, 0.00048, 1, 27)
    genconnect(I3_E, C3, 1, 1, 50)

    #C4
    genconnect(D3_1, C4, 0.00021, 1, 60)
    genconnect(D3_4, C4, 0.00021, 1, 60)
    genconnect(D4_1E, C4, 0.00021, 1, 60)
    genconnect(D4_4E, C4, 0.00021, 1, 60)
    genconnect(I4_E, C4, 1, 1, 50)

    #C5
    genconnect(D5_1, C5, 0.00028, 1, 30)
    genconnect(D5_4, C5, 0.00028, 1, 30)
    genconnect(D4_1E, C5, 0.00028, 1, 30)
    genconnect(D4_4E, C5, 0.00028, 1, 30)
    genconnect(I5_E, C5, 1, 1, 50)

    #C=1 Extensor
    inhgenconnect(C_1, D1_1F, 0.8, 1, 50)
    inhgenconnect(C_1, D1_2F, 0.8, 1, 50)
    inhgenconnect(C_1, D1_3F, 0.8, 1, 50)
    inhgenconnect(C_1, D1_4F, 0.8, 1, 50)

    inhgenconnect(C_1, D2_1F, 0.8, 1, 50)
    inhgenconnect(C_1, D2_2F, 0.8, 1, 50)
    inhgenconnect(C_1, D2_3F, 0.8, 1, 50)
    inhgenconnect(C_1, D2_4F, 0.8, 1, 50)

    inhgenconnect(C_1, D4_1F, 0.8, 1, 50)
    inhgenconnect(C_1, D4_2F, 0.8, 1, 50)
    inhgenconnect(C_1, D4_3F, 0.8, 1, 50)
    inhgenconnect(C_1, D4_4F, 0.8, 1, 50)

    inhgenconnect(C_1, G3_1F, 0.8, 1, 50)
    inhgenconnect(C_1, G3_2F, 0.8, 1, 50)
    inhgenconnect(C_1, G3_3F, 0.8, 1, 50)

    inhgenconnect(C_1, IP1_F, 0.8, 1, 60)
    inhgenconnect(C_1, IP2_F, 0.8, 1, 60)
    inhgenconnect(C_1, IP3_F, 0.8, 1, 60)
    inhgenconnect(C_1, IP4_F, 0.8, 1, 60)
    inhgenconnect(C_1, IP5_F, 0.8, 1, 60)

    inhgenconnect(C_1, Ia_aff_F, 0.9, 1, 80)

    #C=0 Flexor
    inhgenconnect(C_0, IP1_E, 0.8, 1, 60)
    inhgenconnect(C_0, IP2_E, 0.8, 1, 60)
    inhgenconnect(C_0, IP3_E, 0.8, 1, 60)
    inhgenconnect(C_0, IP4_E, 0.8, 1, 60)
    inhgenconnect(C_0, IP5_E, 0.8, 1, 60)

    inhgenconnect(C_0, Ia_aff_E, 0.8, 1, 80)

    #reflex arc
    exconnectcells(Ia_aff_E, Ib_E, 0.00015, 1, 30)
    exconnectcells(Ia_aff_E, Ia_E, 0.01, 1, 30)
    exconnectcells(self.mns_E, R_E, 0.00025, 1, 30)
    inhconnectcells(Ib_E, self.mns_E, 0.01, 1, 45)
    inhconnectcells(Ia_E, self.mns_F, 0.04, 1, 45)
    inhconnectcells(R_E, self.mns_E, 0.05, 1, 45)
    inhconnectcells(R_E, Ia_E, 0.01, 1, 40)

    exconnectcells(Ia_aff_F, Ib_F, 0.00015, 1, 30)
    exconnectcells(Ia_aff_F, Ia_F, 0.01, 1, 30)
    exconnectcells(self.mns_F, R_F, 0.0002, 1, 30)
    inhconnectcells(Ib_F, self.mns_F, 0.05, 1, 45)
    inhconnectcells(Ia_F, self.mns_E, 0.04, 1, 45)
    inhconnectcells(R_F, self.mns_F, 0.01, 1, 45)
    inhconnectcells(R_F, Ia_F, 0.001, 1, 20)

    inhconnectcells(Ib_E, Ib_F, 0.04, 1, 30)
    inhconnectcells(R_E, R_F, 0.04, 1, 30)
    inhconnectcells(Ia_E, Ia_F, 0.04, 1, 30)
    inhconnectcells(Ib_F, Ib_E, 0.04, 1, 30)
    inhconnectcells(R_F, R_E, 0.04, 1, 30)
    inhconnectcells(Ia_F, Ia_E, 0.04, 1, 30)
    
  def addpool(self, num, delaytype):
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = interneuron(delaytype)
      self.interneurons.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)
    return gids

  def addmotoneurons(self, num):
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = motoneuron()
      self.motoneurons.append(cell)
      while(pc.gid_exists(gid)!= 0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)
    return gids

  def addafferents(self, num):
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = bioaff(random.randint(2, 10))
      self.afferents.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)
    return gids

  def addgener(self, start, interval, nums):
    gid = 0
    stim = h.NetStim()
    stim.number = nums
    stim.start = start
    stim.interval = interval
    #skinstim.noise = 0.1
    self.stims.append(stim)
    while(pc.gid_exists(gid)!=0):
        gid+=1
    pc.set_gid2node(gid, rank)
    ncstim = h.NetCon(stim, None)
    pc.cell(gid, ncstim)
    return gid

exnclist = []
inhnclist = []
eesnclist = []
stimnclist = []

def exconnectcells(pre, post, weight, delay, nsyn):
  ''' connection with excitatory synapse '''
  global exnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in post:
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(pre[0], pre[-1])
        target = pc.gid2cell(i)
        syn = target.synlistex[j]
        nc = pc.gid_connect(srcgid, syn)
        exnclist.append(nc)
        nc.delay = random.gauss(delay, delay/6)
        nc.weight[0] = random.gauss(weight, weight/10)

def inhconnectcells(pre, post, weight, delay, nsyn):
  ''' connection with inhibitory synapse '''
  global inhnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in post:
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(pre[0], pre[-1]) 
        target = pc.gid2cell(i)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(srcgid, syn)
        inhnclist.append(nc)
        nc.delay = random.gauss(delay, 0.01)
        nc.weight[0] = random.gauss(weight, weight/10)

def genconnect(afferents_gids, gen_gid, weight, delay, nsyn):
  ''' stimulate afferents with NetStim '''
  global stimnclist
  for i in afferents_gids:
    if pc.gid_exists(i):
      for j in range(nsyn):        
        target = pc.gid2cell(i)
        syn = target.synlistees[j]
        nc = pc.gid_connect(gen_gid, syn)
        stimnclist.append(nc)
        nc.delay = random.gauss(delay, delay/4)
        nc.weight[0] = random.gauss(weight, weight/10)

def inhgenconnect(gen_gid, afferents_gids, weight, delay, nsyn):
  ''' stimulate afferents with NetStim '''
  global stimnclist
  for i in afferents_gids:
    if pc.gid_exists(i):
      for j in range(nsyn):        
        target = pc.gid2cell(i)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(gen_gid, syn)
        stimnclist.append(nc)
        nc.delay = random.gauss(delay, delay/10)
        nc.weight[0] = random.gauss(weight, weight/10)

def connectdelay_extensor(d1, d2, d3, d4):
  exconnectcells(d2, d1, 0.05, 2, 27)
  exconnectcells(d1, d2, 0.05, 2, 27)
  exconnectcells(d1, d3, 0.01, 1, 27)
  exconnectcells(d2, d3, 0.01, 1, 27)
  inhconnectcells(d4, d3, 0.0001, 1, 27)
  inhconnectcells(d3, d2, 0.08, 1, 27)
  inhconnectcells(d3, d1, 0.08, 1, 27)

def connectdelay_flexor(d1, d2, d3, d4):
  exconnectcells(d2, d1, 0.05, 3, 27)
  exconnectcells(d1, d2, 0.05, 3, 27)
  exconnectcells(d2, d3, 0.01, 1, 27)
  exconnectcells(d1, d3, 0.01, 1, 27)
  inhconnectcells(d4, d3, 0.008, 1, 27)
  inhconnectcells(d3, d2, 0.0008, 1, 27)
  inhconnectcells(d3, d1, 0.0008, 1, 27)

def connectgenerator(g1, g2, g3):
  exconnectcells(g1, g2, 0.05, 3, 27)
  exconnectcells(g2, g1, 0.05, 3, 27)
  exconnectcells(g2, g3, 0.005, 1, 27)
  exconnectcells(g1, g3, 0.005, 1, 27)
  inhconnectcells(g3, g1, 0.008, 1, 27)
  inhconnectcells(g3, g2, 0.008, 1, 27)

def connectexpools_extensor(d1, d4, ep):
  exconnectcells(ep, d1, 0.00037, 1, 27)
  exconnectcells(ep, d4, 0.00037, 1, 27)

def spike_record(cpg):
  ''' record spikes from all gids '''
  global moto_EX_v_vec, moto_FL_v_vec, soma_v_vec 
  '''
  for i in range(len(interneurons)):
    v_vec = h.Vector()
    v_vec.record(interneurons[i].soma(0.5)._ref_v)
    soma_v_vec.append(v_vec)
  '''
  for i in cpg.mns_E:
    moto = pc.gid2cell(i)
    moto_vec = h.Vector()
    moto_vec.record(moto.soma(0.5)._ref_vext[0])
    moto_EX_v_vec.append(moto_vec)

  for i in cpg.mns_F:
    moto = pc.gid2cell(i)
    moto_vec = h.Vector()
    moto_vec.record(moto.soma(0.5)._ref_vext[0])
    moto_FL_v_vec.append(moto_vec)

  for i in range(len(cpg.interneurons)):
    v_vec = h.Vector()
    v_vec.record(cpg.interneurons[i].soma(0.5)._ref_v)
    soma_v_vec.append(v_vec)

def prun(speed, cpg):
  ''' simulation control '''
  tstop = 6*speed + 125
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)
  '''
  genconnect(cpg.Ia_aff_E[:30], cpg.ees, 1, 0, 30)
  exconnectcells(cpg.Ia_aff_E[:30], cpg.mns_E, 0.3, 1, 50)
  h.continuerun(h.t+speed)
  genconnect(cpg.Ia_aff_E[:50], cpg.ees, 1, 0, 30)
  exconnectcells(cpg.Ia_aff_E[:50], cpg.mns_E, 0.5, 1, 50)
  h.continuerun(h.t+speed)
  genconnect(cpg.Ia_aff_E, cpg.ees, 1, 0, 50)
  exconnectcells(cpg.Ia_aff_E, cpg.mns_E, 0.7, 1, 50)
  h.continuerun(h.t+speed+10)
  genconnect(cpg.Ia_aff_E[:50], cpg.ees, 0, 0, 30)
  h.continuerun(h.t+2*speed)
  genconnect(cpg.Ia_aff_E[50:80], cpg.ees, 0, 0, 30)
  h.continuerun(h.t+speed) #+125)
  '''

def spikeout(cpg):
  ''' report simulation results to stdout '''
  global rank, moto_EX_v_vec, moto_FL_v_vec, soma_v_vec
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for j in range(len(cpg.mns_E)):
        path=str('./res/vMN_EX%dr%dv%d'%(j, rank, version))
        f = open(path, 'w')
        for v in list(moto_EX_v_vec[j]):
          f.write(str(v)+"\n")
    pc.barrier()

  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for j in range(len(cpg.mns_F)):
        path=str('./res/vMN_FL%dr%dv%d'%(j, rank, version))
        f = open(path, 'w')
        for v in list(moto_FL_v_vec[j]):
          f.write(str(v)+"\n")
    pc.barrier()

  '''
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for j in range(len(cpg.interneurons)):
        path=str('./res/vIn%dr%dv%d'%(j, rank, version))
        f = open(path, 'w')
        for v in list(soma_v_vec[j]):
          f.write(str(v)+"\n")
    pc.barrier()
  '''

def finish():
  ''' proper exit '''
  pc.runworker()
  pc.done()
  h.quit()

if __name__ == '__main__':
  cpg = cpg(speed, EES_i, 100)
  spike_record(cpg)
  print("- "*10, "\nstart")
  prun(speed, cpg)
  print("- "*10, "\nend")
  spikeout(cpg)
  if (nhost > 1):
    finish()
  
