import nest
import pylab


def graph_volts_data_creator(nest, multimeter, neurons_groups, data_type):
    """

    Args:
        nest:
        multimeter:
        neurons_groups:
        data_type: "V_m", "g_in", "g_ex"

    Returns:

    """
    volts_data_moving = []

    dmm = nest.GetStatus(multimeter)[0]
    vms = dmm['events'][data_type]
    ts = dmm["events"]["times"]
    our_data = zip(vms, ts)
    our_data = sorted(our_data, key=lambda x: x[1])

    n_len = len(neurons_groups)
    ts = [i[1] for i in our_data[::n_len]]
    for i in range(0, len(our_data), n_len):  # среднее арифметическое значений
        volts_data_moving.append(sum([i[0] for i in our_data[i:i + n_len]]) / n_len)

    return volts_data_moving, ts


def graph_spike_data_creator(nest, spike_det, neurons_groups):
    spike_time = []

    dSD = nest.GetStatus(spike_det)[0]
    ts = dSD['events']["times"]

    for i in range(len(ts)):
        spike_time.append(i)

    spike_data_moving = [0 for i in spike_time]
    return spike_data_moving, list(ts)


def slice_creator(all_neurons, multimeters, name):
    volts_data_moving, ts = graph_volts_data_creator(nest, multimeters[name], all_neurons[name], "V_m")

    volts_data_moving = [i * -0.03 for i in volts_data_moving]  # Сжимаем в 30 раз по высоте

    new_volts_data_moving = []
    new_ts = []
    shift_num = 2
    for i in range(6):
        new_ts.append(ts[1000 * i:1000 * (i + 1)])
        v_d_m = volts_data_moving[1000 * i:1000 * (i + 1)]

        new_volts_data_moving.append([k + i * shift_num for k in v_d_m])
        pylab.plot(ts, volts_data_moving)

    figure_name = "{0}(sliced)".format(name)
    pylab.figure(figure_name)
    for i in range(6):
        pylab.plot(new_ts[0], new_volts_data_moving[i])
    pylab.savefig("pics(volts)/{0}".format(figure_name), fmt='png')


def plot_pics(all_neurons, multimeters, spike_detectors):
    needed = [
         "OM1_0", "OM1_1", "OM1_2_E", "OM1_2_F", "OM1_3", "iIP_E", "E1",
        # "OM2_0", "OM2_1", "OM2_2_E", "OM2_2_F", "OM2_3", "iIP_E", "E2",
        # "OM3_0", "OM3_1", "OM3_2_E", "OM3_2_F", "OM3_3", "iIP_E", "E3",
        # "OM4_0", "OM4_1", "OM4_2_E", "OM4_2_F", "OM4_3", "iIP_E", "E4",
        # "OM5_0", "OM5_1", "OM5_2_E", "OM5_2_F", "OM5_3", "iIP_E", "E5",
        # "iIP_E", "iIP_F", "eIP_E", "eIP_F",
        # "Ia_E_aff", "Ia_F_aff", "Ia_E_pool", "Ia_F_pool",
        # "MN_E", "MN_F", "R_E", "R_F"
    ]
    # needed = ["OM1_1"]
    for i in all_neurons:
        if i in needed:
            # volts
            volts_data_moving, ts = graph_volts_data_creator(nest, multimeters[i], all_neurons[i], "V_m")
            spike_data_moving, ts2 = graph_spike_data_creator(nest, spike_detectors[i], all_neurons[i])

            figure_name = i + "(volts)"
            pylab.figure(i + "(volts)")
            pylab.plot(ts, volts_data_moving)
            pylab.plot(ts2, spike_data_moving, '.', color='r')
            pylab.savefig("pics(volts)/{0}".format(i), fmt='png')
            pylab.close(figure_name)

            # current
            g_in_data_moving, ts = graph_volts_data_creator(nest, multimeters[i], all_neurons[i], "g_in")
            g_ex_data_moving, ts2 = graph_volts_data_creator(nest, multimeters[i], all_neurons[i], "g_ex")

            figure_name = i + "(in_and_ex)"
            pylab.figure(i + "(in_and_ex)")
            pylab.plot(ts, g_in_data_moving, color='b')
            pylab.plot(ts2, g_ex_data_moving, color='r')
            pylab.savefig("pics(in_and_ex)/{0}".format(i), fmt='png')
            # pylab.show()
            pylab.close(figure_name)

    slice_creator(all_neurons, multimeters, 'MN_E')
    slice_creator(all_neurons, multimeters, 'MN_F')
