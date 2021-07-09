import os
import sys
import pylab as plt
import logging as log
from matplotlib.ticker import MaxNLocator


log.basicConfig(format='[%(funcName)s]: %(message)s', level=log.INFO)
logger = log.getLogger()


def read_data(path):
	names = []
	voltage = []
	g_exc = []
	g_inh = []
	spikes = []

	for filename in filter(lambda f: f.endswith(".dat"), sorted(os.listdir(path))):
		with open(f"{path}/{filename}") as file:
			names.append(filename.replace(".dat", ""))
			voltage.append(list(map(float, file.readline().split())))
			g_exc.append(list(map(float, file.readline().split())))
			g_inh.append(list(map(float, file.readline().split())))
			spikes.append(list(map(float, file.readline().split())))
	logger.info(f"done: {path}")

	return names, voltage, g_exc, g_inh, spikes

import numpy as np

def draw_slice_borders(sim_time, skin_stim_time, extensor_first=True):
	step_cycle = 11 * skin_stim_time

	if extensor_first:
		step_pack = np.array([6 * skin_stim_time, step_cycle])
	else:
		step_pack = np.array([5 * skin_stim_time, step_cycle])

	for _ in range(int(sim_time / step_cycle)):
		plt.axvline(x=step_pack[0], linewidth=3, color='k')
		plt.axvline(x=step_pack[1], linewidth=3, color='k')
		step_pack += step_cycle

	for i in range(0, sim_time, skin_stim_time):
		plt.axvline(x=i, linewidth=1, color='k')

def plot(skin_stim_time, names, voltages, g_exc, g_inh, spikes, step, save_to, plot_only=None):
	for name, voltage, g_e, g_i, s in zip(names, voltages, g_exc, g_inh, spikes):
		if plot_only and name != plot_only:
			continue
		sim_time = int(len(voltage) * step)
		slices_number = int(len(voltage) * step / 25)
		shared_x = list(map(lambda x: x * step, range(len(voltage))))

		if "MN_E" in name or "MN_F" in name:
			plt.figure(figsize=(16, 9))
			plt.suptitle(f"Sliced {name}")
			V_rest = 72

			for slice_index in range(slices_number):
				chunk_start = int(slice_index * 25 / step)
				chunk_end = int((slice_index + 1) * 25 / step)

				Y = [-v - V_rest for v in voltage[chunk_start:chunk_end]]
				X = [x * step for x in range(len(Y))]
				shifted_y = [y + 50 * slice_index for y in Y]

				plt.plot(X, shifted_y, linewidth=0.7, color='r')

			for t in [13, 15, 17, 21]:
				plt.axvline(t, color='gray', linewidth=2, linestyle='--')

			plt.xlim(0, 25)
			plt.yticks([])
			plt.savefig(f"{save_to}/sliced_{name}.png", format="png", dpi=200)
			plt.close()

		plt.figure(figsize=(16, 9))
		plt.suptitle(name)

		# 1
		ax1 = plt.subplot(311)
		draw_slice_borders(sim_time, skin_stim_time)

		plt.plot(shared_x, voltage, color='b')
		plt.plot(s, [0] * len(s), '.', color='r', alpha=0.7)

		plt.xlim(0, sim_time)
		plt.ylim(-100, 60)
		# a small hotfix to hide x lables but save x ticks -- set them as white color
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.ylabel("Voltage, mV")

		plt.grid()

		# 2
		ax2 = plt.subplot(312, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)

		plt.plot(shared_x, g_e, color='r')
		plt.plot(shared_x, g_i, color='b')
		ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
		plt.ylabel("Currents, pA")

		plt.ylim(bottom=0)
		plt.xlim(0, sim_time)
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.grid()

		# 3
		plt.subplot(313, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)

		plt.hist(s, bins=range(sim_time))   # bin is equal to 1ms
		plt.xlim(0, sim_time)
		plt.ylim(bottom=0)
		plt.grid()

		plt.ylabel("Spikes, n")
		plt.ylim(bottom=0)
		ticks = range(0, sim_time + 1, 5 if sim_time <= 275 else 25)
		plt.xticks(ticks, ticks, fontsize=8)

		plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.08)

		plt.savefig(f"{save_to}/{name}.png", format="png", dpi=200)
		plt.close()

		logger.info(name)


def run():
	step = 0.025
	skin_stim_time = 25
	nuclei = None

	if len(sys.argv) == 2:
		path = sys.argv[1]
	elif len(sys.argv) == 3:
		path = sys.argv[1]
		nuclei = sys.argv[2]
	else:
		path = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/dat/"

	plot(skin_stim_time, *read_data(path), step=step, save_to=f"{path}/results/", plot_only=nuclei)


if __name__ == "__main__":
	run()
