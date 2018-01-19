import neucogar.api_kernel as api_kernel
import os
from memristive_spinal_cord.layer2.schemes.mk5.toolkit import Plotter
from memristive_spinal_cord.layer2.schemes.mk5.components.parameters import Paths
from memristive_spinal_cord.layer2.schemes.mk5.components.parameters import Constants

toolkit = Plotter(
    os.path.abspath(os.path.dirname(__file__)),
    Paths.DATA_DIR_NAME.value,
    Paths.FIGURES_DIR_NAME.value
)

api_kernel.SetKernelStatus(
    local_num_threads=Constants.LOCAL_NUM_THREADS.value,
    data_path=Paths.DATA_DIR_NAME.value,
    resolution=Constants.RESOLUTION.value
)

from memristive_spinal_cord.layer2.schemes.mk5.layer2 import Layer2
layer2 = Layer2()

api_kernel.Simulate(Constants.SIMULATION_TIME.value)

toolkit.plot_tier(1, 2, 3, 4, 5, 6)
toolkit.plot_interneuronal_pool(show_results=False, split=True, period=35.)
# toolkit.plot_column(show_results=False, column='3')
# toolkit.plot_column(show_results=False, column='Left')
# toolkit.plot_hidden_layers(1, 2, 3, 4, 5, show_results=False)