import ntpath
import logging
import numpy as np
import pylab as plt
from analysis.functions import auto_prepare_data
from analysis.pearson_correlation import calc_correlation
from analysis.PCA import plot_3D_PCA, split_by_slices, get_lat_amp, get_peaks, calc_boxplots
import operator

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

bar_width = 0.9
k_median = 0
k_fliers_high = 5
k_fliers_low = 6


def plot_slices(extensor_data, flexor_data, latencies, ees_hz, step_size, folder, filename):
	"""
	TODO: add docstring
	Args:
		extensor_data (list): values of extensor motoneurons membrane potential
		flexor_data (list): values of flexor motoneurons membrane potential
		latencies (list or np.ndarray): latencies of poly answers per slice
		ees_hz (int): EES stimulation frequency
		step_size (float): data step
		folder (str): save folder path
		filename (str): name of the future path
	"""
	flexor_data = np.array(flexor_data)
	extensor_data = np.array(extensor_data)
	latencies = (np.array(latencies) / step_size).astype(int)

	# additional properties
	slice_in_ms = 1000 / ees_hz
	slice_in_steps = int(slice_in_ms / step_size)

	# calc boxplot per dot
	e_boxplots_per_iter = np.array([calc_boxplots(dot) for dot in extensor_data.T])
	f_boxplots_per_iter = np.array([calc_boxplots(dot) for dot in flexor_data.T])

	e_splitted_per_slice_boxplots = split_by_slices(e_boxplots_per_iter, slice_in_steps)
	f_splitted_per_slice_boxplots = split_by_slices(f_boxplots_per_iter, slice_in_steps)

	all_splitted_per_slice_boxplots = np.vstack((e_splitted_per_slice_boxplots,
	                                             f_splitted_per_slice_boxplots))

	yticks = []
	slices_number = int((len(extensor_data[0]) + len(flexor_data[0])) / (slice_in_ms / step_size))
	colors = iter(['#287a72', '#f2aa2e', '#472650'] * slices_number)

	e_slices_number = int(len(extensor_data[0]) / (slice_in_ms / step_size))
	plt.subplots(figsize=(16, 9))

	for slice_index, data in enumerate(all_splitted_per_slice_boxplots):
		data += slice_index + (0 if slice_index < e_slices_number else 1)
		shared_x = np.arange(len(data[:, k_fliers_high])) * step_size
		plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color=next(colors), alpha=0.7, zorder=3)
		plt.plot(shared_x, data[:, k_median], color='k', zorder=3)
		yticks.append(data[:, k_median][0])

	lat_x = latencies * step_size
	lat_y = [all_splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat in enumerate(latencies)]
	plt.plot(lat_x, lat_y, linewidth=4, linestyle='--', color='k', zorder=3)
	plt.plot(lat_x, lat_y, '.', markersize=25, color='k', zorder=3)

	xticks = range(int(slice_in_ms) + 1)
	xticklabels = [x if i % 5 == 0 else None for i, x in enumerate(xticks)]
	yticklabels = [None] * slices_number
	slice_indexes = range(1, slices_number + 1)
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		yticklabels[i] = slice_indexes[i]

	plt.xticks(xticks, xticklabels, fontsize=56)
	plt.yticks(yticks, yticklabels, fontsize=56)
	plt.xlim(0, slice_in_ms)
	plt.grid(axis='x', alpha=0.5)

	plt.tight_layout()
	plt.savefig(f"{folder}/{filename}.pdf", dpi=250, format="pdf")
	plt.close()


def recolor(boxplot_elements, color, fill_color):
	"""
	Add colors to bars (setup each element)
	Args:
		boxplot_elements (dict):
			components of the boxplot
		color (str):
			HEX color of outside lines
		fill_color (str):
			HEX color of filling
	"""
	for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
		plt.setp(boxplot_elements[element], color=color, linewidth=3)
	plt.setp(boxplot_elements["fliers"], markeredgecolor=color)
	for patch in boxplot_elements['boxes']:
		patch.set(facecolor=fill_color)


def plot_histograms(amp_per_slice, peaks_per_slice, lat_per_slice,
                    all_data, mono_per_slice, folder, filename, ees_hz, step_size):
	"""
	TODO: add docstring
	Args:
		amp_per_slice (list): amplitudes per slice
		peaks_per_slice (list): number of peaks per slice
		lat_per_slice (list): latencies per slice
		all_data (list of list): data per test run
		mono_per_slice (list): end of mono area per slice
		folder (str): folder path
		filename (str): filename of the future file
		ees_hz (int):
		step_size (float):
	"""
	box_distance = 1.2
	color = "#472650"
	fill_color = "#9D8DA3"
	slices_number = len(lat_per_slice)
	slice_length = int(1000 / ees_hz / step_size)
	slice_indexes = np.array(range(slices_number))

	# calc boxplots per iter
	boxplots_per_iter = np.array([calc_boxplots(dot) for dot in np.array(all_data).T])

	xticks = [x * box_distance for x in slice_indexes]
	# set labels
	xticklabels = [None] * len(slice_indexes)
	human_read = [i + 1 for i in slice_indexes]
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		xticklabels[i] = human_read[i]

	# plot histograms
	for data, title in (amp_per_slice, "amplitudes"), (peaks_per_slice, "peaks"):
		# create subplots
		fig, ax = plt.subplots(figsize=(16, 9))
		# plot amplitudes or peaks
		plt.bar(xticks, data, width=bar_width, color=color, zorder=2)
		# set Y ticks
		yticks = ax.get_yticks()
		human_read = list(yticks)
		yticklabels = [None] * len(yticks)
		for i in [0, -1, int(1 / 3 * len(yticks)), int(2 / 3 * len(yticks))]:
			yticklabels[i] = int(human_read[i]) if human_read[i] >= 10 else f"{human_read[i]:.1f}"
		# plot properties
		plt.grid(axis="y")
		plt.xticks(xticks, xticklabels, fontsize=56)
		plt.yticks(yticks, yticklabels, fontsize=56)
		plt.xlim(-0.5, len(slice_indexes) * box_distance - bar_width / 2)
		plt.tight_layout()
		plt.savefig(f"{folder}/{filename}_{title}.pdf", dpi=250, format="pdf")
		plt.close()

		# log.info(f"Plotted {title} for {filename}")

	# form areas
	splitted_per_slice_boxplots = split_by_slices(boxplots_per_iter, slice_length)
	mono_area = [slice_data[:int(time / step_size)] for time, slice_data in zip(mono_per_slice, splitted_per_slice_boxplots)]
	poly_area = [slice_data[int(time / step_size):] for time, slice_data in zip(lat_per_slice, splitted_per_slice_boxplots)]

	# plot per area
	for data_test_runs, title in (mono_area, "mono"), (poly_area, "poly"):
		area_data = []
		data_test_runs = np.array(data_test_runs)
		# calc diff per slice
		for slice_data in data_test_runs:
			area_data.append(abs(slice_data[:, k_fliers_high] - slice_data[:, k_fliers_low]))

		fig, ax = plt.subplots(figsize=(16, 9))

		fliers = dict(markerfacecolor='k', marker='*', markersize=3)
		# plot latencies
		plt.xticks(fontsize=56)
		plt.yticks(fontsize=56)

		lat_plot = ax.boxplot(area_data, positions=xticks, widths=bar_width, patch_artist=True, flierprops=fliers)
		recolor(lat_plot, color, fill_color)

		yticks = np.array(ax.get_yticks())
		yticks = yticks[yticks >= 0]
		human_read = list(yticks)
		yticklabels = [None] * len(yticks)
		for i in [0, -1, int(1 / 3 * len(yticks)), int(2 / 3 * len(yticks))]:
			if human_read[i] >= 10:
				yticklabels[i] = int(human_read[i])
			else:
				yticklabels[i] = f"{human_read[i]:.1f}"
		# plot properties
		plt.xticks(xticks, xticklabels, fontsize=56)
		plt.yticks(yticks, yticklabels, fontsize=56)
		plt.grid(axis="y")
		plt.xlim(-0.5, len(slice_indexes) * box_distance - bar_width / 2)
		plt.ylim(0, ax.get_yticks()[-1])
		plt.tight_layout()
		plt.savefig(f"{folder}/{filename}_{title}.pdf", dpi=250, format="pdf")
		plt.close()

		# log.info(f"Plotted {title} for {filename}")


def __process_dataset(filepaths, save_to, plot_histogram_flag=False, plot_slices_flag=False, plot_pca_flag=False):
	"""
	ToDo add info
	Args:
		filepaths (list):
		plot_histogram_flag (bool):
		plot_slices_flag (bool):
		plot_pca_flag (bool):
	"""
	all_pack = []
	colors = iter(["#275b78", "#287a72", "#f2aa2e", "#472650", "#a6261d", "#f27c2e", "#2ba7b9"] * 10)

	# process each file
	for filepath in filepaths:
		folder = ntpath.dirname(filepath)
		filename = ntpath.basename(filepath)
		data_label = filename.replace('.hdf5', '')
		# get prepared data, EES frequency and data step size
		e_prepared_data, ees_hz, step_size = auto_prepare_data(folder, filename)
		# process latencies and amplitudes per slice
		e_lat_per_slice, amp_per_slice, mono_per_slice = get_lat_amp(e_prepared_data, ees_hz, step_size)
		# process peaks per slice
		peaks_per_slice = get_peaks(e_prepared_data, e_lat_per_slice, ees_hz=ees_hz, step_size=step_size)
		# form data pack
		coords_meta = (np.stack((e_lat_per_slice, amp_per_slice, peaks_per_slice), axis=1), next(colors), data_label)
		all_pack.append(coords_meta)
		# plot histograms of amplitudes and number of peaks
		if plot_histogram_flag:
			plot_histograms(amp_per_slice, peaks_per_slice, e_lat_per_slice, e_prepared_data, mono_per_slice,
			                folder=folder, filename=filename, ees_hz=ees_hz, step_size=step_size)
		# plot all slices with pattern
		if plot_slices_flag:
			f_prepared_data = auto_prepare_data(folder, filename.replace('_E_', '_F_'))[0]
			f_lat_per_slice = get_lat_amp(f_prepared_data, ees_hz, step_size)[0]
			all_lat_per_slice = np.append(e_lat_per_slice, f_lat_per_slice)
			plot_slices(e_prepared_data, f_prepared_data, all_lat_per_slice,
			            ees_hz=ees_hz, step_size=step_size, folder=folder, filename=filename)
	# plot 3D PCA for each plane
	if plot_pca_flag:
		x1, y1, z1= plot_3D_PCA(all_pack, save_to=save_to)
	return x1, y1, z1


def plot_correlation():
	# FixMe: don't forget to change!
	save_to = "/home/alex/el_test"

	data_a_folder = "/home/alex/el_test"
	data_a_filename = "bio_E_21cms_40Hz_i100_2pedal_no5ht_T"

	data_b_folder = "/home/alex/el_test"
	data_b_filename = "gras_E_21cms_40Hz_i0_2pedal_no5ht_T"

	# get extensor from data
	e_data_a, f_data_a, _ = auto_prepare_data(data_a_folder, data_a_filename, step_size_to=0.1)
	e_data_b, f_data_b, _ = auto_prepare_data(data_b_folder, data_b_filename, step_size_to=0.1)

	e_mono_corr, e_poly_corr = calc_correlation(e_data_a, e_data_b)
	f_mono_corr, f_poly_corr = calc_correlation(f_data_a, f_data_b)


	def __plot_corr(mono_corr, poly_corr, muscle):
		plt.figure(figsize=(16, 9))

		title = f"{data_a_filename.split('_')[0]}_{data_b_filename}"
		box_colors = iter(["#275b78", "#287a72"])
		# plot boxplots
		for i, data in enumerate([poly_corr, mono_corr]):
			color = next(box_colors)
			box = plt.boxplot(data, positions=[i], vert=False, whis=[5, 95], widths=0.7, patch_artist=True)
			# change colors for all elements
			for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
				for patch in box[element]:
					if element == "fliers":
						patch.set_markersize(10)
						patch.set_markerfacecolor(color)
						patch.set_markeredgecolor(None)
					patch.set_color(color)
					patch.set_linewidth(3)
			for patch in box['boxes']:
				patch.set(facecolor=f"{color}55")

		yticks = [0, 1]
		ylabels = ["poly", "mono"]
		plt.yticks(yticks, ylabels)

		plt.xticks(fontsize=56)
		plt.yticks(fontsize=56)
		plt.tight_layout()
		plt.savefig(f"{save_to}/{muscle}_{title}.pdf", format="pdf")

	__plot_corr(e_mono_corr, e_poly_corr, "extensort")
	__plot_corr(f_mono_corr, f_poly_corr, "flexor")


def inresect(a, b):
	return list(set(a) & set(b))


def for_article():
	"""
	TODO: add docstring
	"""
	save_to = '/home/anna/Desktop/res/'

	bio_folder = '/home/anna/PycharmProjects/LAB/memristive-spinal-cord/bio-data/hdf5/toe/4/'
	neuron_folder = ""
	gras_folder = "/home/anna/PycharmProjects/LAB/memristive-spinal-cord/GRAS/hdf5/toe/"
	nest_folder = ""

	compare_pack = [
		f"{bio_folder}/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5",
		f"{gras_folder}/gras_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5",
	]

	# control
	plot_pca_flag = True
	plot_slices_flag = False
	plot_histogram_flag = False

	x1, y1, z1 =\
		__process_dataset(compare_pack, save_to, plot_histogram_flag, plot_slices_flag, plot_pca_flag)

	x1 = x1[:2]
	x1[0] = x1[0][:1]
	x1[1] = x1[1][:1]
	y1 = y1[:2]
	y1[0] = y1[0][:1]
	y1[1] = y1[1][:1]
	z1 = z1[:2]
	z1[0] = z1[0][:1]
	z1[1] = z1[1][:1]

	# x1[0] = x1[0].tolist()
	# x1[1] = x1[1].tolist()
	# y1[0] = y1[0].tolist()
	# y1[1] = y1[1].tolist()
	# z1[0] = z1[0].tolist()
	# z1[1] = z1[1].tolist()

	x1[0] = x1[0].flatten()
	x1[1] = x1[1].flatten()

	y1[0] = y1[0].flatten()
	y1[1] = y1[1].flatten()

	z1[0] = z1[0].flatten()
	z1[1] = z1[1].flatten()
	print("len(x1) = ", len(x1), len(x1[0])) # [3][100][100]
	print("----------")
	print(x1[0])
	print("----------")
	print(x1[1])
	print("----------")

	print("len(y1) = ", len(y1), len(y1[0])) # [3][100][100]
	print("----------")
	print(y1[0])
	print("----------")
	print(y1[1])
	print("----------")

	print("len(z1) = ", len(z1), len(z1[0])) # [3][100][100]
	print("----------")
	print(z1[0])
	print("----------")
	print(z1[1])
	print("----------")

	print("types = ", type(x1[0]))
	print(type(x1[1]))
	print(type(x1))
	x_intersection = inresect(x1[0], x1[1])
	y_intersection = inresect(y1[0], y1[1])
	z_intersection = inresect(z1[0], z1[1])
	print("x_intersection = ", x_intersection)
	print("y_intersection = ", y_intersection)
	print("z_intersection = ", z_intersection)
	raise Exception


def run():
	for_article()
	# plot_correlation()


if __name__ == "__main__":
	run()
