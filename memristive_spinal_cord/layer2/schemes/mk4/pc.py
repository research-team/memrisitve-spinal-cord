from neucogar.Nucleus import Nucleus

from memristive_spinal_cord.layer2.models import Neurotransmitters, ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk4.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk4.components.parameters import Constants, Weights
from memristive_spinal_cord.layer2.schemes.mk4.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.mk4.tier import Tier


class PolysynapticCircuit:
    def __init__(self):
        self.__tiers = [Tier(i+1) for i in range(5)]
        self.__E = [
            Nucleus(nucleus_name='Tier6E0'),
            Nucleus(nucleus_name='Tier6E1')
        ]
        self.__I = Nucleus(nucleus_name='Tier6I0')

        for i in range(2):
            self.__E[i].addSubNucleus(
                neurotransmitter=Neurotransmitters.GLU.value,
                number=Constants.NEURONS_IN_GROUP.value,
                params=Neurons.NEUCOGAR.value
            )
        self.__I.addSubNucleus(
            neurotransmitter=Neurotransmitters.GABA.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

    def set_connections(self):
        for upper_tier, lower_tier in zip(self.__tiers[1:], self.__tiers[:-1]):
            upper_tier.connect(lower_tier=lower_tier)

        self.__E[0].nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.__E[1].nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EE.value[5][0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.__E[1].nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.__E[0].nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EE.value[5][1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.__E[1].nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.__I.nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EI.value[5],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.__I.nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=self.__E[0].nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=-Weights.IE.value[5],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.connect_tier6(self.get_tiers()[4])

    def connect_tier6(self, tier5: Tier):
        self.__E[1].nuclei(Neurotransmitters.GLU.value).connect(
            tier5.get_e(3).nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[5],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        tier5.get_e(0).nuclei(Neurotransmitters.GLU.value).connect(
            self.__E[0].nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[4][2],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        tier5.get_e(4).nuclei(Neurotransmitters.GLU.value).connect(
            self.__E[0].nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[4][0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def connect_multimeters(self):
        for i in range(2):
            self.__E[i].nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.__I.nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()

    def get_input(self): return self.__tiers[0].get_e(0)

    def get_output(self): return [tier.get_e(3) for tier in self.__tiers]

    def get_tiers(self): return self.__tiers

    @staticmethod
    def get_number_of_neurons(): return Tier.get_number_of_neurones() + Constants.NEURONS_IN_GROUP.value * 3