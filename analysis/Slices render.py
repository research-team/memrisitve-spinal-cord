import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, argrelextrema


def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y


def calc_frequency(data, samplerate, save_folder, show=False):
	freq_max = []
	for title, art_data in data.items():
		sampling_frequency = samplerate
		sampling_size = len(art_data)  # get size (length) of the data

		# frequency domain representation
		fourier_transform = np.fft.fft(art_data) / sampling_size  # normalize amplitude
		fourier_transform = abs(fourier_transform[range(int(sampling_size / 2))])  # exclude sampling frequency

		# remove the mirrored part of the FFT
		values = np.arange(int(sampling_size / 2))
		time_period = sampling_size / sampling_frequency
		frequencies = values / time_period

		# cuts frequency and calc max
		mask = (frequencies <= 40) & (frequencies >= 20)
		frequencies = frequencies[mask]
		fourier_transform = fourier_transform[mask]
		# find the maximal frequence
		max_value_index = np.argmax(fourier_transform)
		max_frequency = frequencies[max_value_index]
		freq_max.append(max_frequency)
		# plotting
		plt.figure()
		plt.title('Fourier transform depicting the frequency components')
		plt.plot(frequencies, fourier_transform)
		plt.xlabel('Frequency')
		plt.ylabel('Amplitude')

		# squeeze plot
		plt.tight_layout()
		plt.savefig(f'{save_folder}/Frequency.png', format='png')
		if show:
			plt.show()
		plt.close()

	return freq_max[0]


def read_data(datapath):
	filenames = [name[:-4] for name in os.listdir(f"{datapath}") if name.endswith(".mat")]
	for filename in filenames:
		filename = 'on the right side SS'
		dict_data = sio.loadmat(f'{datapath}/{filename}')
		save_folder = f'{datapath}/render/{filename}'

		raw_data = dict_data['data'][0]
		samplerate = int(dict_data['samplerate'][0][0])

		starts = [int(d[0]) for d in dict_data['datastart']]
		ends = [int(d[0]) for d in dict_data['dataend']]
		titles = dict_data['titles'][:-2]
		arts_titles = dict_data['titles'][-2:]
		dx = 1 / samplerate

		arts = {}
		for t, s, e in zip(arts_titles, starts[-1:], ends[-1:]):
			arts[t] = raw_data[s:e]
		frequency = calc_frequency(data=arts, samplerate=samplerate, save_folder=save_folder, show=True)

		title = list(arts.keys())[0]
		data = list(arts.values())[0]
		zip_start_end = render_art(title=title, data=data, save_folder=save_folder, dx=dx, frequency=frequency,
		                           show=False)

		muscles = {}
		for t, s, e in zip(titles, starts, ends):
			muscles[t] = raw_data[s:e]

		for title, data in muscles.items():
			smoothed_render(title=title, data=data, save_folder=save_folder, dx=dx,
			                zip_start_end=zip_start_end, show=False)


def draw_slices(data, zip_start_end, dx, save_folder, title, show=False):
	shift = 0.1

	if zip_start_end:
		for index, (s, e) in enumerate(zip_start_end, 1):
			d = data[s:e] + shift * index
			xticks = np.arange(len(d)) * dx
			plt.plot(xticks, d)
			plt.ylabel("Voltage")
			plt.xlabel("Time")
			if index % 50 == 0:
				plt.suptitle(f'{title} slices {int(index / 50)} part')
				plt.savefig(f'{save_folder}/{title}_slices_{int(index / 50)}_part.png', format='png')
				if show:
					plt.show()
				plt.close()
		plt.close()


def smoothed_render(title, data, save_folder, dx, zip_start_end, show=False):
	fs = 4000.0
	lowcut = 20.0
	highcut = 1000.0
	plt.suptitle(f'{title}')
	data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
	plt.plot(np.arange(len(data)) * dx, data, color='g')
	plt.ylabel("Voltage")
	plt.xlabel("Time (sec)")

	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
	if show:
		plt.show()
	plt.close()
	draw_slices(data, zip_start_end, dx, save_folder=save_folder, title=title,
	            show=False)


def render_art(title, data, save_folder, dx, frequency, show=False):
	debug = False
	fs = 4000.0
	lowcut = 20.0
	highcut = 1000.0

	try:
		data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
		xticks = np.arange(len(data)) * dx

		"""here is not the best code, but the most short (by NumPy) and stable (as some tests have shown)"""
		# get the 1st derivative to get the highest amplitude of voltage changing
		diff = np.diff(data, n=1)
		# slice in ticks: 1 / freq of stimulation = slice in seconds, divide it by dx and get the number of ticks
		slice_length = 1 / frequency / dx
		# approximate number of slices (157.7 -> 158, 134.1 - > 135)
		slices = int(len(data) / slice_length) + 1

		# get the indices of extrema
		extrema_index = argrelextrema(diff, np.less)[0]
		# get the values of extrema
		extrema_vals = diff[extrema_index]
		# make the 2d presentation of the data (extremum per row)
		extrema = np.stack((extrema_index, extrema_vals), axis=-1)
		# sort by value (from the lowest to the highest)
		extrema.view('i8,f8').sort(order=['f1'], axis=0)

		if debug:
			# plot the all found extrema
			plt.plot(diff)
			plt.plot(extrema[:, 0], extrema[:, 1], '.', color='r')

		# top lowest extrema is exactly what we needed
		err_tolerance = 5  # add more slices to catch all potential extrema
		extrema = extrema[:slices + err_tolerance, :]
		# sort by index
		extrema.view('i8,f8').sort(order=['f0'], axis=0)

		def inside(indices_diff):
			# +/-3 just to be sure that the difference of extrema is approx. close to the slice length (in ticks)
			# use abs() for safety
			return slice_length - 3 <= abs(indices_diff) <= slice_length + 3

		# start from the center, while do not find the properly pair of extrema with approx slice length (by indices)
		start_index = (slices + err_tolerance) // 2
		while not inside(extrema[start_index, 0] - extrema[start_index + 1, 0]):
			start_index += 1

		# check extrema from the center to the right
		index = start_index
		while index < len(extrema) - 1:
			if not inside(extrema[index + 1, 0] - extrema[index, 0]):
				# remove the right extrema if difference is too small
				extrema = np.delete(extrema, index + 1, axis=0)
				# do not increment while we are not confident at distance
				continue
			index += 1
		# check extrema from the center to the left
		index = start_index
		while index > 0:
			# because of deleting at the end, we have to check the index (overflow error)
			if index == len(extrema):
				index -= 1
			if not inside(extrema[index, 0] - extrema[index - 1, 0]):
				# remove the left extrema if difference is too small
				extrema = np.delete(extrema, index - 1, axis=0)
				# do not increment while we are not confident at distance
				continue
			index -= 1

		# check all filtered extrema
		assert all(map(lambda x, y: inside(x - y), extrema[:, 0], extrema[1:, 0]))

		if debug:
			plt.plot(extrema[:, 0], extrema[:, 1], '.', color='b')
			plt.show()

		if debug:
			for index, (start, end) in enumerate(zip(extrema[:, 0].astype(int), extrema[1:, 0].astype(int))):
				d = data[start - 10:end - 10] + (index * 0.001)
				t = np.arange(len(d)) * dx * 1000
				plt.plot(t, d)
			plt.show()

		plt.close()
		plt.suptitle('Art')
		plt.plot(xticks, data, color='g')
		plt.ylabel("Voltage")
		plt.xlabel("Time (sec)")

		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
		plt.savefig(f'{save_folder}/Art_smoothed.png', format='png')
		if show:
			plt.show()
		plt.close()

		zip_start_end = list(zip(extrema[:, 0].astype(int), extrema[1:, 0].astype(int)))

		draw_slices(data, zip_start_end, dx, save_folder=save_folder, title=title,
		            show=False)

		return zip_start_end
	except(Exception):
		print('Error. Check file "Frequency.png"')
		return None

def main():
	path = '/home/b-rain/rhythmic/data'
	folder = '20012021'
	datapath = os.path.join(path, folder)
	read_data(datapath)
	print('Done')


if __name__ == '__main__':
	main()
