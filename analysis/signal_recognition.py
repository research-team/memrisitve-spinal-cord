import os
import sys
import cv2
import time
import h5py
import numpy as np
import scipy.io as sio
from fastkde import fastKDE
from matplotlib.path import Path
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QFont
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QLineEdit, QCheckBox, QComboBox, QMessageBox, QButtonGroup
from PyQt5.QtWidgets import QGridLayout, QFileDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QFrame


class PlotWindow(QMainWindow):
	def __init__(self, parent=None):
		super(PlotWindow, self).__init__(parent)
		self.setWindowTitle("Visualization window")
		self.interactive_dist = None
		self._polygon_points = None
		self._anchor_object = None
		self.parent = parent
		# (1) set up the canvas
		self.fig = Figure(figsize=(8, 8), dpi=100)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self)
		# add axes for plotting
		self.ax1 = self.fig.add_subplot(221)
		self.ax2 = self.fig.add_subplot(222, sharex=self.ax1, sharey=self.ax1)
		self.ax3 = self.fig.add_subplot(223, sharex=self.ax1, sharey=self.ax1)
		self.ax4 = self.fig.add_subplot(224)
		self.ax4.axis('off')
		#
		self.ax1.set_title("Interactive area")
		self.ax2.set_title("Processing")
		self.ax3.set_title("Result output")
		self.ax4.set_title("Debug info")
		# build a colorbar at right of Axis №3
		self.cax = make_axes_locatable(self.ax3).append_axes("right", size="5%", pad="2%")
		self.cax.axis('off')
		#
		self.init_interactive()
		# (2) Create the navigation toolbar, tied to the canvas
		self.mpl_toolbar = NavigationToolbar(self.canvas, self)
		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		layout.addWidget(self.mpl_toolbar)
		widget = QWidget()
		widget.setLayout(layout)
		self.setCentralWidget(widget)
		self.resize(900, 900)
		self.fig.tight_layout()
		self.show()

	def init_interactive(self):
		self._polygon_points = []
		self._line, self._dragging_point, self._anc = None, None, None
		self.canvas.mpl_connect('button_press_event', self.on_click)
		self.canvas.mpl_connect('button_release_event', self.on_release)
		self.canvas.mpl_connect('motion_notify_event', self.on_motion)

	def update_plot(self):
		if not self._polygon_points:
			if self._line:
				self._line.set_data([], [])
		else:
			x, y = zip(*self._polygon_points)
			# Add new plot
			if not self._line:
				(self._line, ) = self.ax1.plot(x, y, "r", marker="o", markersize=8, zorder=10)
			# Update current plot
			else:
				x, y = list(x), list(y)
				self._line.set_data(x + [x[0]], y + [y[0]])
		if self._anchor_object:
			if not self._anc:
				(self._anc, ) = self.ax1.plot(*self._anchor_object, color='r', ms=20, marker='x', mew=3)
			else:
				self._anc.set_data(*self._anchor_object)
		self.fig.tight_layout()
		self.canvas.draw()

	def add_point(self, event):
		self._polygon_points.append((event.xdata, event.ydata))

	def remove_point(self, point):
		if point in self._polygon_points:
			self._polygon_points.remove(point)

	def find_neighbor_point(self, event):
		"""
		Find point around mouse position
		Args:
			event: mouse event object
		Returns:
			tuple: (x, y) if there are any point around mouse else None
		"""
		if self._polygon_points:
			nx, ny = min(self._polygon_points, key=lambda p: np.hypot(event.xdata - p[0], event.ydata - p[1]))
			if np.hypot(event.xdata - nx, event.ydata - ny) < self.interactive_dist:
				return nx, ny
		return None

	@staticmethod
	def isBetween(pA, pB, p0):
		p = pB
		p0A = np.hypot(p0[0] - pA[0], p0[1] - pA[1])
		p0B = np.hypot(p0[0] - pB[0], p0[1] - pB[1])
		if p0A < p0B:
			p = pA
		#
		dotproduct = (p0[0] - pA[0]) * (pB[0] - pA[0]) + (p0[1] - pA[1]) * (pB[1] - pA[1])
		if dotproduct < 0:
			return None, None
		#
		squaredlengthba = (pB[0] - pA[0]) * (pB[0] - pA[0]) + (pB[1] - pA[1]) * (pB[1] - pA[1])
		if dotproduct > squaredlengthba:
			return None, None

		return p0B + p0A, p

	def on_click(self, event):
		"""
		Callback method for mouse click event
		Args:
			event: mouse click event
		"""
		# left click
		if event.inaxes in [self.ax1] and event.button == 1:
			point = self.find_neighbor_point(event)
			p_next = None
			p0 = (event.xdata, event.ydata)
			mind = np.inf
			#
			if len(self._polygon_points) >= 3:
				a = self._polygon_points + [self._polygon_points[0]]
				for p1, p2 in zip(a, a[1:]):
					d, p = self.isBetween(p1, p2, p0)
					if d and d < mind:
						mind = d
						p_next = p2
			if point:
				self._dragging_point = point
			elif p_next:
				self._polygon_points.insert(self._polygon_points.index(p_next), p0)
			else:
				self.add_point(event)
			self.update_plot()
		# mid click
		elif event.inaxes in [self.ax1] and event.button == 2:
			self._polygon_points = []
			self.update_plot()
		elif event.inaxes in [self.ax1] and event.button == 3:
			point = self.find_neighbor_point(event)
			if point:
				self.remove_point(point)
				self.update_plot()
				self.parent.filter_exclude()
			else:
				self._anchor_object = (event.xdata, event.ydata)
				self.update_plot()
		#
		if len(self._polygon_points) > 3:
			self.parent.filter_exclude()

	def on_release(self, event):
		"""
		Callback method for mouse release event
		Args:
			event: mouse event
		"""
		if event.inaxes in [self.ax1] and event.button == 1 and self._dragging_point:
			self._dragging_point = None
			self.update_plot()
			self.parent.filter_exclude()

	def on_motion(self, event):
		"""
		Callback method for mouse motion event
		Args:
			event: mouse event
		"""
		if not self._dragging_point:
			return
		if event.xdata is None or event.ydata is None:
			return
		# get index of the previous dragged point
		index = self._polygon_points.index(self._dragging_point)
		# set new point
		self._dragging_point = (event.xdata, event.ydata)
		# update previous point
		self._polygon_points[index] = self._dragging_point
		self.update_plot()


class Application(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self, parent)
		self.setWindowTitle('Computer Vision Analyzer v3.1 (NcN lab product 2020)')
		self.create_main_frame()
		self.create_status_bar()

		self.data = None
		self.mask_inside = None
		self.current_frame = 0

	def open_file(self, path):
		"""
		Try to read data from files with different types
		Args:
			path (str): file path
		Returns:
			tuple : shape of the data
		"""
		self.variables = {}
		try:
			for varname, vardata in sio.loadmat(path).items():
				# get only 4-dim data
				if len(np.shape(vardata)) == 4:
					self.variables[varname] = vardata[:]
		except NotImplementedError:
			with h5py.File(path, 'r') as file:
				for varname, vardata in file.items():
					# get only 4-dim data
					if len(np.shape(vardata)) == 4:
						self.variables[varname] = vardata[:]
		except Exception:
			QMessageBox.about(self, 'Error', "Could not read the file...")

	def file_dialog(self):
		"""
		Invoke PyQT file dialog with unblocking buttons
		"""
		fname = QFileDialog.getOpenFileName(self, "Open file", '', "MAT file (*.mat)")
		# if exists
		if fname[0]:
			self.box_variable.clear()
			# delete old data if exists
			if self.data is not None:
				del self.data
			self.data = None
			# prepare the data
			self.status_text.setText("Unpack .mat file... Please wait")
			QApplication.processEvents()
			QApplication.processEvents()
			#
			self.filepath = fname[0]
			self.open_file(self.filepath)
			self.status_text.setText(f".mat file is unpacked ({self.filepath})")
			# based on data set the possible variables
			self.box_variable.setEnabled(True)
			for varname in self.variables.keys():
				self.box_variable.addItem(str(varname))

	def reshape_data(self):
		"""
		Transpose the multi-dimensional matrix if need
		"""
		# get new shape as tuple
		new_order = tuple(map(int, self.in_data_reshape.text().split()))
		# reshape data to the new shape
		self.data = np.transpose(self.data, new_order)
		self.im_height, self.im_width, self.total_frames, methods_num = self.data.shape
		self.im_shape = (self.im_height, self.im_width)
		# disable buttons of reshaping
		self.in_data_reshape.setEnabled(False)
		self.btn_reshape_data.setEnabled(False)
		# init frames in GUI form
		self.in_start_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setText(str(self.total_frames - 1))
		# update status
		self.status_text.setText(f"Data was reshaped to {self.data.shape}")
		#
		self.box_method.clear()
		for method_index in range(methods_num):
			self.box_method.addItem(str(method_index))
		#
		self.plot_frame.interactive_dist = max(self.im_width, self.im_height) * 0.1

	def choose_variable(self):
		""" Invoked if text in QComboBox is changed """
		# get the user's choose
		var = self.box_variable.currentText()
		if var != '':
			# get the data by name
			self.data = self.variables[var]
			# meta info
			data_shape = self.data.shape
			str_format = len(data_shape) * '{:<5}'
			self.label_fileinfo.setText(f"Shape: {str_format.format(*data_shape)}\n"
			                            f"Index: {str_format.format(*list(range(4)))}")
			self.label_fileinfo.setFont(QFont("Courier New"))
			self.box_method.clear()
			# unblock buttons
			for obj in [self.btn_save_results, self.btn_loop_draw, self.btn_frame_right,
			            self.btn_frame_left, self.btn_reshape_data, self.in_data_reshape, self.box_method]:
				obj.setEnabled(True)
			self.status_text.setText(f"{var} is chosen {data_shape}")

	def filter_exclude(self):
		"""

		Returns:

		"""
		row, col = np.indices(self.im_shape)
		grid_points = np.vstack((col.ravel(), row.ravel())).T
		p = Path(self.plot_frame._polygon_points)  # make a polygon
		self.mask_inside = p.contains_points(grid_points).reshape(self.im_shape)
		self.true_y, self.true_x = np.where(self.mask_inside)

	def method_onchange(self):
		"""

		Returns:

		"""
		text = self.box_method.currentText()
		if text != '':
			methodic = int(text)
			self.plot_frame.ax1.imshow(np.mean(self.data[:, :, :, methodic], axis=2), zorder=-10, cmap='gray')
			if self.plot_frame._line:
				self.plot_frame.update_plot()
			else:
				self.plot_frame.fig.tight_layout()
				self.plot_frame.canvas.draw()

	@staticmethod
	def polygon_area(coords):
		x, y = coords[:, 0], coords[:, 1]
		return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

	@staticmethod
	def align_coord(coords, border):
		"""

		Args:
			coords:
			border:

		Returns:

		"""
		coords[coords >= border - 0.5] = border - 1
		coords[coords <= 0.5] = 0
		return coords

	def update_draws(self, frame, methodic, static_thresh=None, dynamic_thresh=None,
	                 fullness_need=None, separation=None, anchor=None, reversal=None, save_result=False):
		"""
		Calculate contours and draw them on the axis
		Args:
			frame (int): index of the frame
			methodic (int): index of the method
			static_thresh (float): constant threshold value (None - not using static threshold)
			dynamic_thresh (float): maximal number of fragmentation (None - without fragmentation checking)
			fullness_need (float): level of fullnes inside the IOS (None - without fullness checking)
			separation (tuple): lower/upper values of border (None - without separation)
			anchor (tuple): x,y coordinates of anchor (None - without using an anchor)
			reversal (bool): is reversing of color needs
			save_result (bool): flag for skipping drawing if we want just save results
		Returns:
			tuple: x and y coords of the contour if 'save_result' is true
		"""
		cv_cntrs = None
		fullness = None
		debug_ios = None
		max_contour = None
		# get an original data
		mask_in = self.mask_inside
		if mask_in is None:
			return
		original = self.data[:, :, frame, methodic]
		# normalize data from 0 to 4095 with dynamic borders (min and max). It mades grayscale cmap
		image = np.array(original, copy=True)
		# if normalization:
		# 	a_to, b_to = normalization
		# 	i_min, i_max = image.min(), image.max()
		# 	image = (b_to - a_to) * (image - i_min) / (i_max - i_min) + a_to
		# reverse colors if epilepsy radio button checked
		if reversal:
			image = -image
			original = -original
		if separation:
			mask_in = mask_in & (separation[0] <= image) & (image <= separation[1])
		image[~mask_in] = np.min(image[mask_in])
		# blur the image to smooth very light pixels
		# image = cv2.medianBlur(image, 3)
		# set the dynamic thresh value
		in_mask_image = image[mask_in]
		# first, rude iteration
		morph_kernel = np.ones((3, 3), np.uint8)
		mask = np.zeros(shape=image.shape, dtype='uint8')
		#
		if static_thresh:
			threshold_percent = static_thresh
			thresh_value = np.percentile(in_mask_image, threshold_percent)
			# get coordinates of points which are greater than thresh value
			y, x = np.where(image >= thresh_value)
		else:
			threshold_percent = 99
			# 1st raw loop
			while True:
				thresh_value = np.percentile(in_mask_image, threshold_percent)
				# get coordinates of points which are greater than thresh value
				y, x = np.where(image >= thresh_value)
				# calc raw CV contours to decide -- search contour or not
				tmpmask = np.array(mask, copy=True)
				tmpmask[y, x] = 255
				#
				_, thresh_mask = cv2.threshold(tmpmask, 200, 255, cv2.THRESH_BINARY)
				# transform morphology of the mask
				thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, morph_kernel)
				thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, morph_kernel)
				# get the contour of the mask
				*im2, cv_cntrs, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
				# only if number of CV not so big (more fragmentation -- more confidence that there are no epilepsy contour)
				if len(cv_cntrs) > dynamic_thresh:
					break
				threshold_percent -= 2
				if threshold_percent < 50:
					threshold_percent += 2
					break
			# second, more preciescly iteration
			while True:
				threshold_percent += 0.2
				if threshold_percent > 99.8:
					threshold_percent -= 0.2
					break
				thresh_value = np.percentile(in_mask_image, threshold_percent)
				# get coordinates of points which are greater than thresh value
				y, x = np.where(image >= thresh_value)
				# calc raw CV contours to decide -- search contour or not
				tmpmask = np.array(mask, copy=True)
				tmpmask[y, x] = 255
				#
				_, thresh_mask = cv2.threshold(tmpmask, 200, 255, cv2.THRESH_BINARY)
				# transform morphology of the mask
				thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, morph_kernel)
				thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, morph_kernel)
				# get the contour of the mask
				*im2, cv_cntrs, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
				# only if number of CV not so big (more fragmentation -- more confidence that there are no epilepsy contour)
				if len(cv_cntrs) <= dynamic_thresh:
					break
		#
		if len(x) > 10:
			# get a KDE function values based on found points above XY meshgrid
			PDF, (vx, vy) = fastKDE.pdf(x, y)
			# find the contour by maximal area
			max_cont = max(self.plot_frame.ax3.contour(vx, vy, PDF, levels=1, alpha=0).allsegs[1], key=self.polygon_area)
			# limit coordinates within the border
			x_cont = self.align_coord(max_cont[:, 0], self.im_width)
			y_cont = self.align_coord(max_cont[:, 1], self.im_height)
			max_contour = (x_cont, y_cont)
			# get fullness
			p_poly = Path(np.vstack((x_cont, y_cont)).T)
			inside_points = np.vstack((x, y)).T
			inside_count = np.where(p_poly.contains_points(inside_points))[0].size
			# the area of polygon the same as the count of points inside
			all_count = 0.5 * np.abs(np.dot(x_cont, np.roll(y_cont, 1)) - np.dot(y_cont, np.roll(x_cont, 1)))
			fullness = inside_count / all_count * 100
			#
			if fullness_need and fullness < fullness_need:
				debug_ios = max_contour
				max_contour = None
			if max_contour and anchor and not p_poly.contains_points((anchor,)):
				debug_ios = max_contour
				max_contour = None
		#
		if save_result:
			return max_contour
		else:
			self.current_frame = frame
			self.plot_frame.fig.suptitle(f"Frame {frame}")
			self.plot_frame.ax2.clear()
			self.plot_frame.ax3.clear()
			self.plot_frame.ax4.clear()
			self.plot_frame.ax4.axis('off')
			self.plot_frame.cax.clear()

			self.plot_frame.ax1.set_title("Interactive area")
			self.plot_frame.ax2.set_title("Processing")
			self.plot_frame.ax3.set_title("Result output")
			self.plot_frame.ax4.set_title("Debug info")
			if separation:
				m = (image < separation[0]) | (separation[1] < image)
				image[m] = None
			image[~mask_in] = None
			# Axis 2
			self.plot_frame.ax2.imshow(image)
			self.plot_frame.ax2.plot(x, y, '.', color='r', ms=1)
			if max_contour:
				self.plot_frame.ax2.plot(max_contour[0], max_contour[1], color='r', lw=3)
			if debug_ios:
				self.plot_frame.ax2.plot(debug_ios[0], debug_ios[1], color='r', lw=3, ls='--')
			# Axis 3
			im = self.plot_frame.ax3.imshow(image, cmap='jet')
			self.plot_frame.fig.colorbar(im, cax=self.plot_frame.cax)

			if max_contour:
				self.plot_frame.ax3.plot(max_contour[0], max_contour[1], color='r', lw=3)
			if anchor:
				self.plot_frame.ax2.plot(*anchor, color='w', ms=20, marker='x', mew=3)
				self.plot_frame.ax3.plot(*anchor, color='w', ms=20, marker='x', mew=3)
			# Axis 4
			log = f"Area:\n" \
			      f"     min= {np.min(image[mask_in]):.2f}\n" \
			      f"     max= {np.max(image[mask_in]):.2f}\n" \
			      f"Image (original):\n" \
			      f"     min= {np.min(original):.2f}\n" \
			      f"     max= {np.max(original):.2f}\n" \
			      f"Threshold: {threshold_percent:.1f} ({'Static' if static_thresh else 'Dynamic'})\n" \
			      f"Fragments: {len(cv_cntrs) if cv_cntrs else 0} (Max: {dynamic_thresh if dynamic_thresh else 'not setted'})\n" \
			      f"Fullness: {fullness:.2f}% (Min: {round(fullness_need, 2) if fullness_need else 'not setted'})\n"
			if self.chkbox_anchor.isChecked():
				anc = self.plot_frame._anchor_object
				log += f"Anchor: {np.floor(anc) if anc and self.chkbox_anchor.isChecked() else 'Not used'}"
			self.plot_frame.ax4.text(0, 0.5, log, ha='left', va='center', transform=self.plot_frame.ax4.transAxes)
			# save axis plot
			# extent = self.ax3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
			# self.fig.savefig(f'/home/alex/example/{frame}.jpg', format='jpg')
			self.plot_frame.fig.tight_layout()
			self.plot_frame.canvas.draw()
			# waiting to see changes
			time.sleep(0.01)
			# flush the changes to the screen
			self.plot_frame.canvas.flush_events()

	def check_input(self, input_value, borders=(-np.inf, np.inf)):
		"""
		Checking the input value on validity
		Returns:
			float : converted from string a value
		Raises:
			ValueError : value not in borders
			Exception : cannot convert from string
		"""
		try:
			value = float(input_value.text())
			if borders[0] <= value <= borders[1]:
				return value
			else:
				QMessageBox.about(self, f"Error value '{value}'", f"Value must be a number from {borders[0]} to {borders[1]}")
				return
		except Exception:
			QMessageBox.about(self, f"Error value '{input_value.text()}'", f"Value must be a number from {borders[0]} to {borders[1]}")
		return

	def check_fields(self):
		"""
		"""
		fields = {}
		#
		if len(self.plot_frame._polygon_points) < 4:
			QMessageBox.about(self, "Error", f"Polygon is not setted")
			fields['polygon_existing'] = False
		#
		if self.radio_static.isChecked():
			static_thresh = self.check_input(self.in_static_thresh, borders=[0.1, 99.9])
			fields['static_thresh'] = static_thresh
		else:
			dynamic_thresh = self.check_input(self.in_dynamic_thresh, borders=[3, 100])
			fields['dynamic_thresh'] = int(dynamic_thresh) if dynamic_thresh else None
		#
		if self.chkbox_fullness.isChecked():
			fullness_need = self.check_input(self.in_fullness, borders=[0.0, 100.0])
			fields['fullness_need'] = fullness_need
		#
		if self.chkbox_separation.isChecked():
			lower = self.check_input(self.in_lower_separation)
			upper = self.check_input(self.in_upper_separation)
			fields['separation'] = None if (lower is None or upper is None or upper <= lower) else (lower, upper)
		#
		if self.chkbox_anchor.isChecked() and self.plot_frame._anchor_object:
			fields['anchor'] = self.plot_frame._anchor_object
		#
		fields['reversal'] = self.chkbox_reverse.isChecked()

		return fields

	def save_contour(self):
		"""
		Converting numpy arrays of contours to a mat file
		"""
		start = int(self.check_input(self.in_start_frame))
		end = int(self.check_input(self.in_end_frame))
		step = int(self.check_input(self.in_frame_stepsize))
		methodic = int(self.box_method.currentText())
		#
		if start < 0 or end <= start or end >= self.total_frames or step <= 0:
			QMessageBox.about(self, "Error", f"Invalid start/end/step values")
			return
		#
		fields = self.check_fields()
		if any(f in [None, False] for f in fields):
			return
		static_thresh = fields['static_thresh'] if 'static_thresh' in fields.keys() else None
		dynamic_thresh = fields['dynamic_thresh'] if 'dynamic_thresh' in fields.keys() else None
		fullness_need = fields['fullness_need'] if 'fullness_need' in fields.keys() else None
		separation = fields['separation'] if 'separation' in fields.keys() else None
		anchor = fields['anchor'] if 'anchor' in fields.keys() else None
		reversal = fields['reversal']
		# check if value is correct
		self.status_text.setText("Saving results.... please wait")
		# prepare array of objects per frame
		matframes = np.zeros((self.total_frames, ), dtype=np.object)
		# init by void arrays
		for frame in range(self.total_frames):
			matframes[frame] = np.array([], dtype=np.int32)
		# get data per frame and fill the 'matframes'
		for index, frame in enumerate(range(start, end, step)):
			contour = self.update_draws(frame, methodic,
			                            static_thresh=static_thresh, dynamic_thresh=dynamic_thresh,
			                            fullness_need=fullness_need, separation=separation,
			                            anchor=anchor, reversal=reversal, save_result=True)
			if contour is not None:
				matframes[frame] = np.array(contour, dtype=np.int32)
			QApplication.processEvents()
			QApplication.processEvents()
			self.status_text.setText(f"Processed {index / len(range(start, end, step)) * 100:.2f} %")
		# save data into mat format
		filepath = os.path.dirname(self.filepath)
		filename = os.path.basename(self.filepath)[:-4]
		newpath = f"{filepath}/{filename}_{self.box_variable.currentText()}_{methodic}.mat"
		fields['frames'] = matframes
		sio.savemat(newpath, fields)
		# you are beautiful :3
		self.status_text.setText(f"Successfully saved into {newpath}")

	def on_loop_draw(self):
		"""
		Automatic drawing data in loop by user panel settings
		"""
		start = int(self.check_input(self.in_start_frame))
		end = int(self.check_input(self.in_end_frame))
		step = int(self.check_input(self.in_frame_stepsize))
		methodic = int(self.box_method.currentText())
		#
		if start < 0 or end <= start or end >= self.total_frames or step <= 0:
			QMessageBox.about(self, "Error", f"Invalid start/end/step values")
			return
		#
		fields = self.check_fields()
		if any(f in [None, False] for f in fields):
			return
		static_thresh = fields['static_thresh'] if 'static_thresh' in fields.keys() else None
		dynamic_thresh = fields['dynamic_thresh'] if 'dynamic_thresh' in fields.keys() else None
		fullness_need = fields['fullness_need'] if 'fullness_need' in fields.keys() else None
		separation = fields['separation'] if 'separation' in fields.keys() else None
		anchor = fields['anchor'] if 'anchor' in fields.keys() else None
		reversal = fields['reversal']
		#
		self.flag_loop_draw_stop = False
		self.btn_loop_draw_stop.setEnabled(True)

		for frame in range(start, end, step):
			if self.flag_loop_draw_stop:
				break
			self.current_frame = frame
			self.in_start_frame.setText(str(frame))
			self.update_draws(frame, methodic,
			                  static_thresh=static_thresh, dynamic_thresh=dynamic_thresh, fullness_need=fullness_need,
			                  separation=separation, anchor=anchor, reversal=reversal)

	def stop_loop(self):
		self.flag_loop_draw_stop = True

	def on_hand_draw(self, step, sign=1):
		"""
		Manual drawing frames
		Args:
			step (int): stepsize of left/right moving
			sign (int): -1 or 1 show the side moving (-1 is left, 1 is right)
		"""
		self.current_frame += sign * step
		methodic = int(self.box_method.currentText())
		#
		fields = self.check_fields()
		if any(f in [None, False] for f in fields):
			return
		static_thresh = fields['static_thresh'] if 'static_thresh' in fields.keys() else None
		dynamic_thresh = fields['dynamic_thresh'] if 'dynamic_thresh' in fields.keys() else None
		fullness_need = fields['fullness_need'] if 'fullness_need' in fields.keys() else None
		separation = fields['separation'] if 'separation' in fields.keys() else None
		anchor = fields['anchor'] if 'anchor' in fields.keys() else None
		reversal = fields['reversal']
		#
		if self.current_frame < 0:
			self.current_frame = 0
		if self.current_frame >= self.total_frames:
			self.current_frame = self.total_frames - 1

		self.in_start_frame.setText(str(self.current_frame))
		self.update_draws(self.current_frame, methodic,
		                  static_thresh=static_thresh, dynamic_thresh=dynamic_thresh, fullness_need=fullness_need,
		                  separation=separation, anchor=anchor, reversal=reversal)

	def create_main_frame(self):
		# create the main plot
		self.main_frame = QWidget()
		self.plot_frame = PlotWindow(self)
		# (3) Layout with panel
		btn_panel_grid = QGridLayout()
		btn_panel_grid.setContentsMargins(0, 0, 0, 0)

		current_line = 1

		''' PREPARE BLOCK '''
		# FILE
		self.btn_file = QPushButton("Open file")
		self.btn_file.clicked.connect(self.file_dialog)
		btn_panel_grid.addWidget(self.btn_file, current_line, 0, 1, 1)

		label_variable = QLabel("Variable:")
		label_variable.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
		btn_panel_grid.addWidget(label_variable, current_line, 1, 1, 1)
		# VARIABLE
		self.box_variable = QComboBox(self)
		btn_panel_grid.addWidget(self.box_variable, current_line, 2, 1, 1)
		self.box_variable.currentTextChanged.connect(lambda x: self.choose_variable())
		self.box_variable.setEnabled(False)

		current_line += 1

		self.label_fileinfo = QLabel("File info")
		self.label_fileinfo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_fileinfo, current_line, 0, 1, 3)

		current_line += 1

		# RESHAPE
		lbl_reshape_meta = QLabel("Height Width Frame Method")
		lbl_reshape_meta.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(lbl_reshape_meta, current_line, 0, 1, 3)

		current_line += 1

		lbl_reshape = QLabel("Reshape data")
		lbl_reshape.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		btn_panel_grid.addWidget(lbl_reshape, current_line, 0, 1, 1)

		self.in_data_reshape = QLineEdit()
		self.in_data_reshape.setText("0 1 2 3")
		btn_panel_grid.addWidget(self.in_data_reshape, current_line, 1, 1, 1)
		self.in_data_reshape.setEnabled(False)

		self.btn_reshape_data = QPushButton("Reshape")
		self.btn_reshape_data.clicked.connect(lambda x: self.reshape_data())
		btn_panel_grid.addWidget(self.btn_reshape_data, current_line, 2, 1, 1)
		self.btn_reshape_data.setEnabled(False)

		current_line += 1

		# METHOD
		label_method = QLabel("Method")
		label_method.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		btn_panel_grid.addWidget(label_method, current_line, 0, 1, 1)

		self.box_method = QComboBox(self)
		btn_panel_grid.addWidget(self.box_method, current_line, 1, 1, 1)
		self.box_method.setEnabled(False)
		self.box_method.currentTextChanged.connect(self.method_onchange)  # changed!

		current_line += 1

		# Threshold
		def state0():
			if self.radio_static.isChecked():
				self.in_static_thresh.setEnabled(True)
				self.in_dynamic_thresh.setEnabled(False)
			if self.radio_dynamic.isChecked():
				self.in_static_thresh.setEnabled(False)
				self.in_dynamic_thresh.setEnabled(True)

		self.label_object = QLabel("Threshold")
		self.label_object.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_object, current_line, 0, 1, 1)

		radio_thresh_group = QButtonGroup(self.main_frame)
		self.radio_static = QRadioButton("Static")
		self.radio_dynamic = QRadioButton("Dynamic")
		self.radio_static.clicked.connect(state0)
		self.radio_dynamic.clicked.connect(state0)
		radio_thresh_group.addButton(self.radio_static)
		radio_thresh_group.addButton(self.radio_dynamic)
		self.radio_static.setChecked(True)
		btn_panel_grid.addWidget(self.radio_static, current_line, 2, 1, 1)
		btn_panel_grid.addWidget(self.radio_dynamic, current_line + 1, 2, 1, 1)

		self.in_static_thresh = QLineEdit()
		self.in_static_thresh.setPlaceholderText("0 - 100%")
		btn_panel_grid.addWidget(self.in_static_thresh, current_line, 1, 1, 1)

		self.in_dynamic_thresh = QLineEdit()
		self.in_dynamic_thresh.setPlaceholderText("3 - 50 frags")
		self.in_dynamic_thresh.setEnabled(False)
		btn_panel_grid.addWidget(self.in_dynamic_thresh, current_line + 1, 1, 1, 1)

		current_line += 2

		def state1():
			if self.chkbox_separation.isChecked():
				self.in_lower_separation.setEnabled(True)
				self.in_upper_separation.setEnabled(True)
			else:
				self.in_lower_separation.setEnabled(False)
				self.in_upper_separation.setEnabled(False)

		# normalization
		self.chkbox_separation = QCheckBox("Separation")
		self.chkbox_separation.setChecked(False)
		btn_panel_grid.addWidget(self.chkbox_separation, current_line, 0, 1, 1)
		self.chkbox_separation.stateChanged.connect(state1)
		# min
		self.in_lower_separation = QLineEdit()
		self.in_lower_separation.setPlaceholderText("lower")
		self.in_lower_separation.setEnabled(False)
		btn_panel_grid.addWidget(self.in_lower_separation, current_line, 1, 1, 1)
		# max
		self.in_upper_separation = QLineEdit()
		self.in_upper_separation.setPlaceholderText("upper")
		self.in_upper_separation.setEnabled(False)
		btn_panel_grid.addWidget(self.in_upper_separation, current_line, 2, 1, 1)

		current_line += 1

		def state2():
			if self.chkbox_fullness.isChecked():
				self.in_fullness.setEnabled(True)
			else:
				self.in_fullness.setEnabled(False)

		self.chkbox_fullness = QCheckBox("Use fullness")
		self.chkbox_fullness.setChecked(False)
		btn_panel_grid.addWidget(self.chkbox_fullness, current_line, 0, 1, 1)
		self.chkbox_fullness.stateChanged.connect(state2)

		self.in_fullness = QLineEdit()
		self.in_fullness.setPlaceholderText("0 - 100%")
		self.in_fullness.setEnabled(False)
		btn_panel_grid.addWidget(self.in_fullness, current_line, 1, 1, 1)

		current_line += 1

		self.chkbox_reverse = QCheckBox(f"Reverse colors")
		self.chkbox_reverse.setChecked(False)
		btn_panel_grid.addWidget(self.chkbox_reverse, current_line, 0, 1, 1)

		current_line += 1

		self.chkbox_anchor = QCheckBox(f"Use anchor")
		self.chkbox_anchor.setChecked(False)
		btn_panel_grid.addWidget(self.chkbox_anchor, current_line, 0, 1, 1)

		self.line = QFrame()
		self.line.setFrameShape(QFrame.VLine)
		self.line.setFrameShadow(QFrame.Sunken)
		btn_panel_grid.addWidget(self.line, 0, 3, current_line + 2, 1)
		''' END PREPARE BLOCK '''

		current_line = 1

		''' MANUAL BLOCK '''
		self.lbl_manual = QLabel("Manual view")
		self.lbl_manual.setAlignment(Qt.AlignCenter)
		btn_panel_grid.addWidget(self.lbl_manual, current_line, 4, 1, 3)

		current_line += 1

		in_frame_step = QLineEdit("1")
		in_frame_step.setAlignment(Qt.AlignCenter)
		in_frame_step.setValidator(QIntValidator(1, 100))
		btn_panel_grid.addWidget(in_frame_step, current_line, 5, 1, 1)

		left_step = lambda x: self.on_hand_draw(int(self.check_input(in_frame_step)), sign=-1)
		right_step = lambda x: self.on_hand_draw(int(self.check_input(in_frame_step)))

		self.btn_frame_left = QPushButton("<<")
		self.btn_frame_left.clicked.connect(left_step)
		self.btn_frame_left.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_frame_left, current_line, 4, 1, 1)

		self.btn_frame_right = QPushButton(">>")
		self.btn_frame_right.clicked.connect(right_step)
		self.btn_frame_right.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_frame_right, current_line, 6, 1, 1)

		current_line += 1

		self.lbl_framestep = QLabel("Frame step")
		self.lbl_framestep.setAlignment(Qt.AlignCenter)
		btn_panel_grid.addWidget(self.lbl_framestep, current_line, 4, 1, 3)
		''' END MANUAL BLOCK '''

		current_line += 1

		self.line = QFrame()
		self.line.setFrameShape(QFrame.HLine)
		self.line.setFrameShadow(QFrame.Sunken)
		btn_panel_grid.addWidget(self.line, current_line, 4, 1, 3)

		current_line += 1

		''' BEGIN AUTO BLOCK '''
		self.label_automatic = QLabel("Automatic view")
		self.label_automatic.setAlignment(Qt.AlignCenter)
		btn_panel_grid.addWidget(self.label_automatic, current_line, 4, 1, 3)
		current_line += 1
		#
		self.label_start_frame = QLabel("Start frame")
		self.label_start_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_start_frame, current_line, 4, 1, 1)

		self.in_start_frame = QLineEdit("0")
		btn_panel_grid.addWidget(self.in_start_frame, current_line, 6, 1, 1)
		current_line += 1

		self.label_end_frame = QLabel("End frame")
		self.label_end_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_end_frame, current_line, 4, 1, 1)

		self.in_end_frame = QLineEdit("0")
		btn_panel_grid.addWidget(self.in_end_frame, current_line, 6, 1, 1)

		current_line += 1

		self.label_stepsize_frame = QLabel("Step size frame")
		self.label_stepsize_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_stepsize_frame, current_line, 4, 1, 1)

		self.in_frame_stepsize = QLineEdit("1")
		self.in_frame_stepsize.setValidator(QIntValidator(0, 100))
		btn_panel_grid.addWidget(self.in_frame_stepsize, current_line, 6, 1, 1)

		current_line += 1

		self.btn_loop_draw = QPushButton("Start loop draw")
		self.btn_loop_draw.clicked.connect(lambda x: self.on_loop_draw())
		self.btn_loop_draw.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_loop_draw, current_line, 4, 1, 3)

		current_line += 1

		self.btn_loop_draw_stop = QPushButton("Stop loop draw")
		self.btn_loop_draw_stop.clicked.connect(lambda x: self.stop_loop())
		self.btn_loop_draw_stop.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_loop_draw_stop, current_line, 4, 1, 3)

		current_line += 2

		self.btn_save_results = QPushButton("Save results")
		self.btn_save_results.clicked.connect(lambda x: self.save_contour())
		self.btn_save_results.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_save_results, current_line, 4, 1, 3)
		""" END AUTO BLOCK """

		# (4) combne all in the structure
		vbox = QVBoxLayout()
		vbox.addLayout(btn_panel_grid)

		self.main_frame.setLayout(vbox)
		self.setCentralWidget(self.main_frame)

	def create_status_bar(self):
		self.status_text = QLabel("Waiting a file...")
		self.statusBar().addWidget(self.status_text, stretch=1)


def main():
	app = QApplication(sys.argv)
	form = Application()
	form.resize(700, 350)

	form.show()
	app.exec_()

if __name__ == "__main__":
	main()
