import pylab as plt
import logging as log

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotting')

nrns_id_start = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
                 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 996, 1165, 1185, 1205, 1225,
                 1245, 1265, 1285, 1305, 1325, 1520]

groups_name = ["C1", "C2", "C3", "C4", "C5", "D1_1", "D1_2", "D1_3", "D1_4", "D2_1", "D2_2",
              "D2_3", "D2_4", "D3_1", "D3_2", "D3_3", "D3_4", "D4_1", "D4_2", "D4_3", "D4_4",
              "D5_1", "D5_2", "D5_3", "D5_4", "G1_1", "G1_2", "G1_3", "G2_1", "G2_2", "G2_3",
              "G3_1", "G3_2", "G3_3", "G4_1", "G4_2", "G4_3", "G5_1", "G5_2", "G5_3", "IP_E",
              "MP_E", "EES", "inh_group3", "inh_group4", "inh_group5", "ees_group1", "ees_group2",
              "ees_group3", "ees_group4", "Ia"]


def read_data(path):
	data_container = {}
	with open(path) as file:
		for line in file:
			gid, *data = line.split()
			data_container[int(gid)] = [float(d) for d in data]
	logger.info("done : {}".format(path))
	return data_container


def process_data(data, form_in_group):
	logger.info("Start processing...")
	if form_in_group:
		combined_data = {}

		for index in range(len(nrns_id_start) - 1):
			neuron_number = nrns_id_start[index + 1] - nrns_id_start[index]
			group = groups_name[index]

			if group not in combined_data.keys():
				combined_data[group] = [v / neuron_number for v in data[index]]

			for nrn_id in range(nrns_id_start[index], nrns_id_start[index + 1]):
				combined_data[group] = [a + b / neuron_number for a, b in zip(combined_data[group], data[nrn_id])]
		return combined_data
	return data


def plot(volt, curr, step, save_to):
	for v, i in zip(volt.items(), curr.items()):
		title = v[0]
		voltages = v[1]
		currents = i[1]

		plt.figure(figsize=(10, 5))
		plt.suptitle(title)
		ax1 = plt.subplot(211)
		plt.plot([x * step for x in range(len(voltages))], voltages, color='b', label='voltages')
		for slice_index in range(6):
			plt.axvline(x=slice_index * 25, color='grey', linestyle='--')
		plt.legend()
		plt.xlim(0, len(voltages) * step)

		plt.subplot(212, sharex=ax1)
		plt.plot([x * step for x in range(len(currents))], currents, color='r', label='currents')
		for slice_index in range(6):
			plt.axvline(x=slice_index * 25, color='grey', linestyle='--')
		plt.legend()
		plt.xlim(0, len(voltages) * step)

		filename = "{}.png".format(title)

		plt.savefig("{}/{}".format(save_to, filename), format="png")
		plt.close()

		logger.info(title)


def run():
	step = 0.25
	form_in_group = True
	save_to = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU"
	filepath_volt = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/volt.dat"
	filepath_curr = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/curr.dat"

	neurons_volt = read_data(filepath_volt)
	neurons_curr = read_data(filepath_curr)

	neurons_volt = process_data(neurons_volt, form_in_group=form_in_group)
	neurons_curr = process_data(neurons_curr, form_in_group=form_in_group)

	plot(neurons_volt, neurons_curr, step=step, save_to=save_to)


if __name__ == "__main__":
	run()

