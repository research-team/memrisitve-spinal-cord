import nest
import definitions
from memristive_spinal_cord.proposed_scheme.neuron_network import NeuronNetwork
import memristive_spinal_cord.util as util

from memristive_spinal_cord.proposed_scheme.moraud.params.original.afferents import afferent_params
from memristive_spinal_cord.proposed_scheme.moraud.params.original.neuron_groups import neuron_group_params
from memristive_spinal_cord.proposed_scheme.moraud.params.original.devices import device_params
from memristive_spinal_cord.proposed_scheme.moraud.params.original.connections import connection_params_list
from memristive_spinal_cord.proposed_scheme.moraud.entities import Layer1Multimeters
from memristive_spinal_cord.proposed_scheme.results_plotter import ResultsPlotter
import memristive_spinal_cord.proposed_scheme.device_data as device_data

nest.SetKernelStatus({"total_num_virtual_procs": 8,
                      "print_time": True})

def plot_neuron_group(flexor_device: Layer1Multimeters, extensor_device: Layer1Multimeters, group_name: str) -> None:

    flexor_motor_data = device_data.get_average_voltage(
        flexor_device,
        definitions.RESULTS_DIR
    )
    extensor_motor_data = device_data.get_average_voltage(
        extensor_device,
        definitions.RESULTS_DIR
    )
    plotter.subplot(flexor_motor_data, extensor_motor_data, group_name)


nest.Install("research_team_models")

entity_params = dict()
entity_params.update(afferent_params)
entity_params.update(neuron_group_params)
entity_params.update(device_params)

util.clean_previous_results()
layer1 = NeuronNetwork(entity_params, connection_params_list)

nest.Simulate(10000.)

plotter = ResultsPlotter(3, 'Layer1 average "V_m" of neuron groups')
plotter.reset()
plot_neuron_group(Layer1Multimeters.FLEX_MOTOR, Layer1Multimeters.EXTENS_MOTOR, 'Motor')
plot_neuron_group(Layer1Multimeters.FLEX_INTER_2, Layer1Multimeters.EXTENS_INTER_2, 'Inter2')
plot_neuron_group(Layer1Multimeters.FLEX_INTER_1A, Layer1Multimeters.EXTENS_INTER_1A, 'Inter1A')
plotter.show()