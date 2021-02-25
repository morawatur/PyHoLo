# Copyright (C) 2020  Krzysztof Morawiec
#
# This file is part of PyHoLo.
#
# PyHoLo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHoLo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHoLo.  If not, see <https://www.gnu.org/licenses/>.

#-------------------------------------------------------------------

import re
import sys
from os import path
from functools import partial
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
import Dm3Reader as dm3
import Constants as const
import ImageSupport as imsup
import Transform as tr
import Holography as holo
import MagCalc as mc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# --------------------------------------------------------

class RgbColorTable:
    def __init__(self):
        step = 6
        inc_range = np.arange(0, 256, step)
        dec_range = np.arange(255, -1, -step)
        bcm1 = [QtGui.qRgb(0, i, 255) for i in inc_range]
        gcm1 = [QtGui.qRgb(0, 255, i) for i in dec_range]
        gcm2 = [QtGui.qRgb(i, 255, 0) for i in inc_range]
        rcm1 = [QtGui.qRgb(255, i, 0) for i in dec_range]
        rcm2 = [QtGui.qRgb(255, 0, i) for i in inc_range]
        bcm2 = [QtGui.qRgb(i, 0, 255) for i in dec_range]
        self.cm = bcm1 + gcm1 + gcm2 + rcm1 + rcm2 + bcm2

# --------------------------------------------------------

class RgbColorTable_B2R:
    def __init__(self):
        step = 4
        inc_range = np.arange(0, 256, step)
        dec_range = np.arange(255, -1, -step)
        bcm1 = [QtGui.qRgb(0, i, 255) for i in inc_range]
        gcm1 = [QtGui.qRgb(0, 255, i) for i in dec_range]
        gcm2 = [QtGui.qRgb(i, 255, 0) for i in inc_range]
        rcm1 = [QtGui.qRgb(255, i, 0) for i in dec_range]
        # rcm2 = [QtGui.qRgb(255, 0, i) for i in inc_range]
        # bcm2 = [QtGui.qRgb(i, 0, 255) for i in dec_range]
        self.cm = bcm1 + gcm1 + gcm2 + rcm1

# --------------------------------------------------------

class SimpleImageLabel(QtWidgets.QLabel):
    def __init__(self, image=None):
        super(SimpleImageLabel, self).__init__()
        self.image = image
        self.set_image()

    def set_image(self, disp_amp=True):
        if disp_amp:
            px_arr = np.copy(self.image.amph.am)
        else:
            px_arr = np.copy(self.image.amph.ph)

        pixmap = imsup.scale_image(px_arr, 0.0, 255.0)
        q_image = QtGui.QImage(pixmap.astype(np.uint8), pixmap.shape[0], pixmap.shape[1], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(q_image)
        self.setPixmap(pixmap)
        self.repaint()

# --------------------------------------------------------

class LabelExt(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        blank_image = imsup.ImageExp(const.disp_dim, const.disp_dim, num=-1)
        self.image = image if image is not None else blank_image
        self.set_image()
        self.show_lines = True
        self.show_labs = True
        self.rgb_cm = RgbColorTable_B2R()

    def paintEvent(self, event):
        super(LabelExt, self).paintEvent(event)
        line_pen = QtGui.QPen(QtCore.Qt.yellow)
        line_pen.setCapStyle(QtCore.Qt.RoundCap)
        line_pen.setWidth(3)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        img_idx = abs(self.image.num_in_ser) - 1
        points = self.parent().point_sets[img_idx]
        qp.setPen(line_pen)
        qp.setBrush(QtCore.Qt.yellow)

        for pt in points:
            # rect = QtCore.QRect(pt[0]-3, pt[1]-3, 7, 7)
            # qp.drawArc(rect, 0, 16*360)
            qp.drawEllipse(pt[0]-3, pt[1]-3, 7, 7)

        line_pen.setWidth(2)
        if self.show_lines:
            qp.setPen(line_pen)
            for pt1, pt2 in zip(points, points[1:] + points[:1]):
                line = QtCore.QLine(pt1[0], pt1[1], pt2[0], pt2[1])
                qp.drawLine(line)

        line_pen.setStyle(QtCore.Qt.DashLine)
        line_pen.setColor(QtCore.Qt.yellow)
        line_pen.setCapStyle(QtCore.Qt.FlatCap)
        qp.setPen(line_pen)
        qp.setBrush(QtCore.Qt.NoBrush)
        if len(points) == 2:
            pt1, pt2 = points
            pt1, pt2 = tr.convert_points_to_tl_br(pt1, pt2)
            w = np.abs(pt2[0] - pt1[0])
            h = np.abs(pt2[1] - pt1[1])
            rect = QtCore.QRect(pt1[0], pt1[1], w, h)
            qp.drawRect(rect)
            sq_coords = imsup.make_square_coords(pt1 + pt2)
            sq_pt1 = sq_coords[:2]
            sq_pt2 = sq_coords[2:]
            w = np.abs(sq_pt2[0]-sq_pt1[0])
            h = np.abs(sq_pt2[1]-sq_pt1[1])
            square = QtCore.QRect(sq_pt1[0], sq_pt1[1], w, h)
            line_pen.setColor(QtCore.Qt.red)
            qp.setPen(line_pen)
            qp.drawRect(square)
        qp.end()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        curr_pos = [pos.x(), pos.y()]
        img_idx = abs(self.image.num_in_ser) - 1
        points = self.parent().point_sets[img_idx]
        points.append(curr_pos)
        self.repaint()

        pt_idx = len(points)
        real_x, real_y = disp_pt_to_real_tl_pt(self.image.width, curr_pos)
        print('Added point {0} at:\nx = {1}\ny = {2}'.format(pt_idx, pos.x(), pos.y()))
        print('Actual position:\nx = {0}\ny = {1}'.format(real_x, real_y))
        print('Amp = {0:.2f}\nPhs = {1:.2f}'.format(self.image.amph.am[real_y, real_x], self.image.amph.ph[real_y, real_x]))

        if self.show_labs:
            lab = QtWidgets.QLabel('{0}'.format(pt_idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pos.x()+4, pos.y()+4)
            lab.show()

    def set_image(self, disp_amp=True, disp_phs=False, log_scale=False, hide_bad_px=False, color=False, update_bcg=False, bright=0, cont=255, gamma=1.0):
        if self.image.buffer.am.shape[0] != const.disp_dim:
            self.image = rescale_image_buffer_to_window(self.image, const.disp_dim)

        if disp_amp:
            px_arr = np.copy(self.image.buffer.am)
            if hide_bad_px:
                imsup.remove_outlier_pixels_ip(px_arr, const.min_px_threshold, const.max_px_threshold)
            if log_scale:
                imsup.prep_arr_and_calc_log_ip(px_arr)
        elif disp_phs:
            px_arr = np.copy(self.image.buffer.ph)
        else:
            if self.image.cos_phase is None: self.image.update_cos_phase()
            px_arr = np.cos(self.image.buffer.ph)

        if not update_bcg:
            pixmap_to_disp = imsup.scale_image(px_arr, 0.0, 255.0)
        else:
            pixmap_to_disp = imsup.update_image_bright_cont_gamma(px_arr, brg=bright, cnt=cont, gam=gamma)

        # final image with all properties set
        q_image = QtGui.QImage(pixmap_to_disp.astype(np.uint8), pixmap_to_disp.shape[0], pixmap_to_disp.shape[1],
                               QtGui.QImage.Format_Indexed8)

        if color:
            q_image.setColorTable(self.rgb_cm.cm)

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.disp_dim)
        self.setPixmap(pixmap)
        self.repaint()

    def hide_labels(self):
        labs_to_del = self.children()
        for child in labs_to_del:
            child.deleteLater()

    def show_labels(self):
        img_idx = self.image.num_in_ser - 1
        points = self.parent().point_sets[img_idx]
        n_pt = len(points)
        for pt, idx in zip(points, range(1, n_pt+1)):
            lab = QtWidgets.QLabel('{0}'.format(idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pt[0] + 4, pt[1] + 4)
            lab.show()

    def show_last_label(self):
        img_idx = abs(self.image.num_in_ser) - 1
        points = self.parent().point_sets[img_idx]
        pt_idx = len(points) - 1
        last_pt = points[pt_idx]
        lab = QtWidgets.QLabel('{0}'.format(pt_idx+1), self)
        lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
        lab.move(last_pt[0] + 4, last_pt[1] + 4)
        lab.show()

    def update_labels(self):
        self.hide_labels()
        if self.show_labs:
            self.show_labels()

# --------------------------------------------------------

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.marked_points = []
        self.marked_points_data = []
        self.canvas.mpl_connect('button_press_event', self.get_xy_data_on_click)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, data_x, data_y, xlab='x', ylab='y'):
        self.figure.clf()
        self.marked_points = []
        self.marked_points_data = []
        ax = self.figure.add_subplot(111)
        ax.plot(data_x, data_y, '.-')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.axis([min(data_x) - 0.5, max(data_x) + 0.5, min(data_y) - 0.5, max(data_y) + 0.5])
        self.canvas.draw()

    def get_xy_data_on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if len(self.marked_points) == 2:
            for pt in self.marked_points:
                pt.remove()
            self.marked_points = []
            self.marked_points_data = []
        ax = self.figure.axes[0]
        pt, = ax.plot(event.xdata, event.ydata, 'ro')
        print('x={0:.2f}, y={1:.2f}'.format(event.xdata, event.ydata))
        self.marked_points.append(pt)
        self.marked_points_data.append([event.xdata, event.ydata])
        self.canvas.draw()

# --------------------------------------------------------

class LineEditWithLabel(QtWidgets.QWidget):
    def __init__(self, parent, lab_text='Label', default_input=''):
        super(LineEditWithLabel, self).__init__(parent)
        self.label = QtWidgets.QLabel(lab_text, self)
        self.input = QtWidgets.QLineEdit(default_input, self)
        self.initUI()

    def initUI(self):
        self.input.setMaxLength(10)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.label)
        vbox.addWidget(self.input)
        self.setLayout(vbox)

# --------------------------------------------------------

class ImgScrollArea(QtWidgets.QScrollArea):
    def __init__(self, any_img=None):
        super(ImgScrollArea, self).__init__()

        self.scroll_content = QtWidgets.QWidget(self)
        self.scroll_layout = QtWidgets.QHBoxLayout(self.scroll_content)
        self.scroll_content.setLayout(self.scroll_layout)

        if any_img is not None:
            self.update_scroll_list(any_img)

        self.setWidget(self.scroll_content)

    def update_scroll_list(self, any_img):
        n_items = self.scroll_layout.count()
        if n_items > 0:
            for i in reversed(range(n_items)):
                self.scroll_layout.itemAt(i).widget().deleteLater()

        first_img = imsup.get_first_image(any_img)
        img_list = imsup.create_image_list_from_first_image(first_img)
        if len(img_list) > 0:
            for img in img_list:
                preview_img = create_preview_img(img, (64, 64))
                preview = SimpleImageLabel(preview_img)
                self.scroll_layout.addWidget(preview)

# --------------------------------------------------------

def create_preview_img(full_img, new_sz):
    sx, sy = new_sz
    preview = imsup.ImageExp(sx, sy, full_img.cmp)
    preview.amph.am = np.copy(full_img.amph.am[:sx, :sy])
    preview.amph.ph = np.copy(full_img.amph.ph[:sx, :sy])
    return preview

# --------------------------------------------------------

class HolographyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(HolographyWindow, self).__init__()
        self.holo_widget = HolographyWidget()

        # ------------------------------
        # Menu bar
        # ------------------------------

        open_img_act = QtWidgets.QAction('Open dm3 or npy...', self)
        open_img_act.setShortcut('Ctrl+O')
        open_img_act.triggered.connect(self.open_image_files)

        open_img_ser_act = QtWidgets.QAction('Open dm3 series...', self)
        open_img_ser_act.setShortcut('Ctrl+D')
        open_img_ser_act.triggered.connect(self.open_image_series)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(open_img_act)
        file_menu.addAction(open_img_ser_act)

        # ------------------------------

        self.statusBar().showMessage('')

        self.setCentralWidget(self.holo_widget)

        self.move(250, 50)
        self.setWindowTitle('Holo window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def open_image_files(self):
        import pathlib
        curr_dir = str(pathlib.Path().absolute())

        file_dialog = QtWidgets.QFileDialog()
        file_paths = file_dialog.getOpenFileNames(self, 'Open image files', curr_dir, 'Image files (*.dm3 *.npy)')[0]

        if len(file_paths) == 0:
            print('No images to read...')
            return

        img_type = 'amp' if self.holo_widget.amp_radio_button.isChecked() else 'phs'

        self.show_status_bar_message('Reading files...', change_bkg=True)

        for fpath in file_paths:
            print('Reading file "{0}"'.format(fpath))

            # dm3 file
            if fpath.endswith('.dm3'):
                img_name_match = re.search('(.+)/(.+).dm3$', fpath)
                img_name_text = img_name_match.group(2)

                new_img = open_dm3_file(fpath, img_type)
            # npy file
            elif fpath.endswith('.npy'):
                img_name_match = re.search('(.+)/(.+).npy$', fpath)
                img_name_text = img_name_match.group(2)

                new_img_arr = np.load(fpath)
                h, w = new_img_arr.shape

                new_img = imsup.ImageExp(h, w, imsup.Image.cmp['CAP'])
                if img_type == 'amp':
                    new_img.load_amp_data(new_img_arr)
                else:
                    new_img.load_phs_data(new_img_arr)
            else:
                print('Could not load the image. It must be in dm3 or npy format...')
                return

            # in the case of npy file the px_dim will be the same as for the last dm3 file opened
            new_img.name = img_name_text
            new_img = rescale_image_buffer_to_window(new_img, const.disp_dim)

            self.holo_widget.insert_img_after_curr(new_img)

            if not self.holo_widget.tab_disp.isEnabled():
                self.holo_widget.enable_tabs()

        self.show_status_bar_message('', change_bkg=True)

    def open_image_series(self):
        import pathlib
        curr_dir = str(pathlib.Path().absolute())

        file_dialog = QtWidgets.QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Open dm3 series', curr_dir, 'Image files (*.dm3)')[0]

        if file_path == '':
            print('No images to read...')
            return
        if not file_path.endswith('.dm3'):
            print('Could not load the starting image. It must be in dm3 format...')
            return

        self.show_status_bar_message('Reading files...', change_bkg=True)

        first_img = load_image_series_from_first_file(file_path)
        if first_img is None:
            self.show_status_bar_message('', change_bkg=True)
            return

        self.holo_widget.insert_img_after_curr(first_img)

        if not self.holo_widget.tab_disp.isEnabled():
            self.holo_widget.enable_tabs()

        self.show_status_bar_message('', change_bkg=True)

    def show_status_bar_message(self, msg='', change_bkg=False):
        status_bar = self.statusBar()
        status_bar.showMessage(msg)
        if not change_bkg:
            return
        if status_bar.styleSheet() == '':
            status_bar.setStyleSheet('background-color : yellow')
        else:
            status_bar.setStyleSheet('')
        status_bar.repaint()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_A:
            self.holo_widget.go_to_prev_image()
        elif event.key() == QtCore.Qt.Key_D:
            self.holo_widget.go_to_next_image()

# --------------------------------------------------------

class HolographyWidget(QtWidgets.QWidget):
    def __init__(self):
        super(HolographyWidget, self).__init__()
        self.display = LabelExt(self)
        self.display.setFixedWidth(const.disp_dim)
        self.display.setFixedHeight(const.disp_dim)
        self.display.setStyleSheet('background-color: black;')
        self.plot_widget = PlotWidget()
        self.preview_scroll = ImgScrollArea()
        self.backup_image = None
        self.point_sets = [[]]
        self.last_shift = [0, 0]     # [dx, dy]
        self.last_rot_angle = 0
        self.last_scale_factor = 1.0
        self.warp_points = []
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(350)

        self.curr_info_label = QtWidgets.QLabel('Image info', self)
        self.only_int = QtGui.QIntValidator()

        # ------------------------------
        # Navigation panel (1)
        # ------------------------------

        prev_button = QtWidgets.QPushButton('<---', self)
        next_button = QtWidgets.QPushButton('--->', self)
        first_button = QtWidgets.QPushButton('<<<-', self)
        last_button = QtWidgets.QPushButton('->>>', self)
        lswap_button = QtWidgets.QPushButton('L-Swap', self)
        rswap_button = QtWidgets.QPushButton('R-Swap', self)
        make_first_button = QtWidgets.QPushButton('Make first', self)
        make_last_button = QtWidgets.QPushButton('Make last')
        set_name_button = QtWidgets.QPushButton('Set name', self)
        reset_names_button = QtWidgets.QPushButton('Reset names', self)
        delete_button = QtWidgets.QPushButton('Delete', self)
        clear_button = QtWidgets.QPushButton('Clear', self)
        rm_last_marker_button = QtWidgets.QPushButton('Remove last marker', self)
        add_marker_at_xy_button = QtWidgets.QPushButton('Add marker', self)
        transfer_phs_to_img_button = QtWidgets.QPushButton('Transfer current phase to image No. -->', self)

        self.amp_radio_button = QtWidgets.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtWidgets.QRadioButton('Phase', self)
        self.cos_phs_radio_button = QtWidgets.QRadioButton('Phase cosine', self)
        self.amp_radio_button.setChecked(True)

        amp_phs_group = QtWidgets.QButtonGroup(self)
        amp_phs_group.addButton(self.amp_radio_button)
        amp_phs_group.addButton(self.phs_radio_button)
        amp_phs_group.addButton(self.cos_phs_radio_button)

        self.name_input = QtWidgets.QLineEdit('', self)

        marker_xy_label = QtWidgets.QLabel('Marker xy-coords:')
        self.marker_x_input = QtWidgets.QLineEdit('0', self)
        self.marker_y_input = QtWidgets.QLineEdit('0', self)

        self.img_num_for_phs_transfer_input = QtWidgets.QLineEdit('1', self)

        prev_button.clicked.connect(self.go_to_prev_image)
        next_button.clicked.connect(self.go_to_next_image)
        first_button.clicked.connect(self.go_to_first_image)
        last_button.clicked.connect(self.go_to_last_image)
        lswap_button.clicked.connect(self.swap_left)
        rswap_button.clicked.connect(self.swap_right)
        make_first_button.clicked.connect(self.make_image_first)
        make_last_button.clicked.connect(self.make_image_last)
        set_name_button.clicked.connect(self.set_image_name)
        reset_names_button.clicked.connect(self.reset_image_names)
        delete_button.clicked.connect(self.delete_image)
        clear_button.clicked.connect(self.clear_image)
        rm_last_marker_button.clicked.connect(self.remove_last_marker)
        add_marker_at_xy_button.clicked.connect(self.add_marker_at_xy)
        transfer_phs_to_img_button.clicked.connect(self.transfer_phase_to_image)

        self.amp_radio_button.toggled.connect(self.update_display_and_bcg)
        self.phs_radio_button.toggled.connect(self.update_display_and_bcg)
        self.cos_phs_radio_button.toggled.connect(self.update_display_and_bcg)

        self.tab_nav = QtWidgets.QWidget(self)
        self.tab_nav.layout = QtWidgets.QGridLayout()
        self.tab_nav.layout.setColumnStretch(0, 1)
        self.tab_nav.layout.setColumnStretch(1, 1)
        self.tab_nav.layout.setColumnStretch(2, 1)
        self.tab_nav.layout.setColumnStretch(3, 1)
        self.tab_nav.layout.setColumnStretch(4, 1)
        self.tab_nav.layout.setColumnStretch(5, 1)
        self.tab_nav.layout.setRowStretch(0, 1)
        self.tab_nav.layout.setRowStretch(9, 1)
        self.tab_nav.layout.addWidget(first_button, 1, 1)
        self.tab_nav.layout.addWidget(prev_button, 1, 2)
        self.tab_nav.layout.addWidget(next_button, 1, 3)
        self.tab_nav.layout.addWidget(last_button, 1, 4)
        self.tab_nav.layout.addWidget(make_first_button, 2, 1)
        self.tab_nav.layout.addWidget(lswap_button, 2, 2)
        self.tab_nav.layout.addWidget(rswap_button, 2, 3)
        self.tab_nav.layout.addWidget(make_last_button, 2, 4)
        self.tab_nav.layout.addWidget(self.name_input, 3, 1, 1, 2)
        self.tab_nav.layout.addWidget(set_name_button, 4, 1)
        self.tab_nav.layout.addWidget(reset_names_button, 4, 2)
        self.tab_nav.layout.addWidget(clear_button, 3, 3, 1, 2)
        self.tab_nav.layout.addWidget(delete_button, 4, 3, 1, 2)
        self.tab_nav.layout.addWidget(rm_last_marker_button, 5, 3, 1, 2)
        self.tab_nav.layout.addWidget(self.amp_radio_button, 5, 1)
        self.tab_nav.layout.addWidget(self.phs_radio_button, 6, 1)
        self.tab_nav.layout.addWidget(self.cos_phs_radio_button, 7, 1)
        self.tab_nav.layout.addWidget(marker_xy_label, 6, 3)
        self.tab_nav.layout.addWidget(add_marker_at_xy_button, 7, 3)
        self.tab_nav.layout.addWidget(self.marker_x_input, 6, 4)
        self.tab_nav.layout.addWidget(self.marker_y_input, 7, 4)
        self.tab_nav.layout.addWidget(transfer_phs_to_img_button, 8, 1, 1, 3)
        self.tab_nav.layout.addWidget(self.img_num_for_phs_transfer_input, 8, 4)
        self.tab_nav.setLayout(self.tab_nav.layout)

        # ------------------------------
        # Display panel (2)
        # ------------------------------

        unwrap_button = QtWidgets.QPushButton('Unwrap', self)
        wrap_button = QtWidgets.QPushButton('Wrap', self)
        norm_phase_button = QtWidgets.QPushButton('Norm. phase', self)
        crop_button = QtWidgets.QPushButton('Crop N ROIs', self)
        flip_h_button = QtWidgets.QPushButton('Flip H', self)
        flip_v_button = QtWidgets.QPushButton('Flip V', self)
        blank_area_button = QtWidgets.QPushButton('Blank area', self)
        export_button = QtWidgets.QPushButton('Export', self)
        export_all_button = QtWidgets.QPushButton('Export all', self)

        self.show_lines_checkbox = QtWidgets.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtWidgets.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.log_scale_checkbox = QtWidgets.QCheckBox('Log scale', self)
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_display_and_bcg)

        self.hide_bad_px_checkbox = QtWidgets.QCheckBox('Hide bad pixels', self)
        self.hide_bad_px_checkbox.setChecked(False)
        self.hide_bad_px_checkbox.toggled.connect(self.update_display_and_bcg)

        self.clear_prev_checkbox = QtWidgets.QCheckBox('Clear prev. images', self)
        self.clear_prev_checkbox.setChecked(False)

        self.gray_radio_button = QtWidgets.QRadioButton('Grayscale', self)
        self.color_radio_button = QtWidgets.QRadioButton('Color', self)
        self.gray_radio_button.setChecked(True)

        color_group = QtWidgets.QButtonGroup(self)
        color_group.addButton(self.gray_radio_button)
        color_group.addButton(self.color_radio_button)

        self.export_tiff_radio_button = QtWidgets.QRadioButton('TIFF image', self)
        self.export_npy_radio_button = QtWidgets.QRadioButton('Numpy array file', self)
        self.export_raw_radio_button = QtWidgets.QRadioButton('Raw data file', self)
        self.export_tiff_radio_button.setChecked(True)

        export_group = QtWidgets.QButtonGroup(self)
        export_group.addButton(self.export_tiff_radio_button)
        export_group.addButton(self.export_npy_radio_button)
        export_group.addButton(self.export_raw_radio_button)

        self.n_to_crop_input = QtWidgets.QLineEdit('1', self)

        fname_label = QtWidgets.QLabel('File name', self)
        self.fname_input = QtWidgets.QLineEdit('', self)

        unwrap_button.clicked.connect(self.unwrap_phase)
        wrap_button.clicked.connect(self.wrap_phase)
        norm_phase_button.clicked.connect(self.norm_phase)
        crop_button.clicked.connect(self.crop_n_fragments)
        flip_h_button.clicked.connect(partial(self.flip_image, True))
        flip_v_button.clicked.connect(partial(self.flip_image, False))
        blank_area_button.clicked.connect(self.blank_area)
        export_button.clicked.connect(self.export_image)
        export_all_button.clicked.connect(self.export_all)

        self.gray_radio_button.toggled.connect(self.update_display_and_bcg)
        self.color_radio_button.toggled.connect(self.update_display_and_bcg)

        grid_disp = QtWidgets.QGridLayout()
        grid_disp.setColumnStretch(0, 1)
        grid_disp.setColumnStretch(1, 1)
        grid_disp.setColumnStretch(2, 1)
        grid_disp.setColumnStretch(3, 1)
        grid_disp.setColumnStretch(4, 1)
        grid_disp.setColumnStretch(5, 1)
        grid_disp.setRowStretch(0, 1)
        grid_disp.setRowStretch(5, 1)
        grid_disp.addWidget(self.show_lines_checkbox, 1, 1)
        grid_disp.addWidget(self.show_labels_checkbox, 2, 1)
        grid_disp.addWidget(self.log_scale_checkbox, 3, 1)
        grid_disp.addWidget(self.gray_radio_button, 1, 2)
        grid_disp.addWidget(self.color_radio_button, 2, 2)
        grid_disp.addWidget(self.hide_bad_px_checkbox, 3, 2)
        grid_disp.addWidget(unwrap_button, 1, 3)
        grid_disp.addWidget(wrap_button, 2, 3)
        grid_disp.addWidget(norm_phase_button, 3, 3)
        grid_disp.addWidget(crop_button, 4, 1)
        grid_disp.addWidget(self.n_to_crop_input, 4, 2)
        grid_disp.addWidget(self.clear_prev_checkbox, 4, 3, 1, 2)
        grid_disp.addWidget(flip_h_button, 1, 4)
        grid_disp.addWidget(flip_v_button, 2, 4)
        grid_disp.addWidget(blank_area_button, 3, 4)

        grid_exp = QtWidgets.QGridLayout()
        grid_exp.setColumnStretch(0, 1)
        grid_exp.setColumnStretch(1, 1)
        grid_exp.setColumnStretch(2, 1)
        grid_exp.setColumnStretch(3, 1)
        grid_exp.setColumnStretch(4, 1)
        grid_exp.setColumnStretch(5, 1)
        grid_exp.setRowStretch(0, 1)
        grid_exp.setRowStretch(4, 1)
        grid_exp.addWidget(fname_label, 1, 1, 1, 2)
        grid_exp.addWidget(self.fname_input, 2, 1, 1, 2)
        grid_exp.addWidget(export_button, 3, 1)
        grid_exp.addWidget(export_all_button, 3, 2)
        grid_exp.addWidget(self.export_tiff_radio_button, 1, 3, 1, 2)
        grid_exp.addWidget(self.export_npy_radio_button, 2, 3, 1, 2)
        grid_exp.addWidget(self.export_raw_radio_button, 3, 3, 1, 2)

        self.tab_disp = QtWidgets.QWidget(self)
        self.tab_disp.layout = QtWidgets.QVBoxLayout()
        self.tab_disp.layout.addLayout(grid_disp)
        self.tab_disp.layout.addLayout(grid_exp)
        self.tab_disp.setLayout(self.tab_disp.layout)

        # ------------------------------
        # Manual alignment panel (3)
        # ------------------------------

        self.move_left_button = QtWidgets.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        self.move_right_button = QtWidgets.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        self.move_up_button = QtWidgets.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        self.move_down_button = QtWidgets.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        self.rot_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_right.png'), '', self)
        self.rot_counter_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_left.png'), '', self)
        self.apply_button = QtWidgets.QPushButton('Apply changes', self)
        self.reset_button = QtWidgets.QPushButton('Reset', self)

        self.manual_mode_checkbox = QtWidgets.QCheckBox('Manual mode', self)
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.clicked.connect(self.create_backup_image)

        self.px_shift_input = QtWidgets.QLineEdit('0', self)
        self.rot_angle_input = QtWidgets.QLineEdit('0.0', self)

        self.move_left_button.clicked.connect(self.move_left)
        self.move_right_button.clicked.connect(self.move_right)
        self.move_up_button.clicked.connect(self.move_up)
        self.move_down_button.clicked.connect(self.move_down)
        self.rot_clockwise_button.clicked.connect(self.rotate_right)
        self.rot_counter_clockwise_button.clicked.connect(self.rotate_left)
        self.apply_button.clicked.connect(self.apply_changes)
        self.reset_button.clicked.connect(partial(self.reset_changes, True))

        self.disable_manual_panel()

        # ------------------------------
        # Automatic alignment panel (4)
        # ------------------------------

        auto_shift_button = QtWidgets.QPushButton('Auto-Shift', self)
        auto_rot_button = QtWidgets.QPushButton('Auto-Rotate', self)
        get_scale_ratio_button = QtWidgets.QPushButton('Get scale ratio from image calib.')
        scale_button = QtWidgets.QPushButton('Scale', self)
        warp_button = QtWidgets.QPushButton('Warp', self)
        reshift_button = QtWidgets.QPushButton('Re-Shift', self)
        rerot_button = QtWidgets.QPushButton('Re-Rotate', self)
        rescale_button = QtWidgets.QPushButton('Re-Scale', self)
        rewarp_button = QtWidgets.QPushButton('Re-Warp', self)
        cross_corr_w_prev_button = QtWidgets.QPushButton('Cross corr. w. prev.', self)
        cross_corr_all_button = QtWidgets.QPushButton('Cross corr. all', self)

        self.scale_factor_input = QtWidgets.QLineEdit('1.0', self)

        auto_shift_button.clicked.connect(self.auto_shift_image)
        auto_rot_button.clicked.connect(self.auto_rotate_image)
        get_scale_ratio_button.clicked.connect(self.get_scale_ratio_from_images)
        scale_button.clicked.connect(self.scale_image)
        warp_button.clicked.connect(partial(self.warp_image, False))
        reshift_button.clicked.connect(self.reshift)
        rerot_button.clicked.connect(self.rerotate)
        rescale_button.clicked.connect(self.rescale_image)
        rewarp_button.clicked.connect(self.rewarp)
        cross_corr_w_prev_button.clicked.connect(self.cross_corr_with_prev)
        cross_corr_all_button.clicked.connect(self.cross_corr_all)

        cross_corr_w_prev_button.setEnabled(False)
        cross_corr_all_button.setEnabled(False)

        grid_manual = QtWidgets.QGridLayout()
        grid_manual.setColumnStretch(0, 1)
        grid_manual.setColumnStretch(1, 2)
        grid_manual.setColumnStretch(2, 2)
        grid_manual.setColumnStretch(3, 2)
        grid_manual.setColumnStretch(4, 2)
        grid_manual.setColumnStretch(5, 2)
        grid_manual.setColumnStretch(6, 2)
        grid_manual.setColumnStretch(7, 2)
        grid_manual.setColumnStretch(8, 1)
        grid_manual.setRowStretch(0, 1)
        grid_manual.setRowStretch(4, 1)
        grid_manual.addWidget(self.move_left_button, 2, 1)
        grid_manual.addWidget(self.move_right_button, 2, 3)
        grid_manual.addWidget(self.move_up_button, 1, 2)
        grid_manual.addWidget(self.move_down_button, 3, 2)
        grid_manual.addWidget(self.px_shift_input, 2, 2)
        grid_manual.addWidget(self.manual_mode_checkbox, 1, 4)
        grid_manual.addWidget(self.apply_button, 2, 4)
        grid_manual.addWidget(self.reset_button, 3, 4)
        grid_manual.addWidget(self.rot_counter_clockwise_button, 2, 5)
        grid_manual.addWidget(self.rot_angle_input, 2, 6)
        grid_manual.addWidget(self.rot_clockwise_button, 2, 7)

        grid_auto = QtWidgets.QGridLayout()
        grid_auto.setColumnStretch(0, 1)
        grid_auto.setColumnStretch(1, 2)
        grid_auto.setColumnStretch(2, 2)
        grid_auto.setColumnStretch(3, 2)
        grid_auto.setColumnStretch(4, 1)
        grid_auto.setRowStretch(0, 1)
        grid_auto.setRowStretch(6, 1)
        grid_auto.addWidget(auto_shift_button, 1, 1)
        grid_auto.addWidget(auto_rot_button, 2, 1)
        grid_auto.addWidget(get_scale_ratio_button, 3, 1, 1, 2)
        grid_auto.addWidget(scale_button, 4, 1)
        grid_auto.addWidget(warp_button, 5, 1)
        grid_auto.addWidget(reshift_button, 1, 2)
        grid_auto.addWidget(rerot_button, 2, 2)
        grid_auto.addWidget(rescale_button, 4, 2)
        grid_auto.addWidget(rewarp_button, 5, 2)
        grid_auto.addWidget(cross_corr_w_prev_button, 1, 3)
        grid_auto.addWidget(cross_corr_all_button, 2, 3)
        grid_auto.addWidget(self.scale_factor_input, 4, 3)

        self.tab_align = QtWidgets.QWidget(self)
        self.tab_align.layout = QtWidgets.QVBoxLayout()
        self.tab_align.layout.addLayout(grid_manual)
        self.tab_align.layout.addLayout(grid_auto)
        self.tab_align.setLayout(self.tab_align.layout)

        # ------------------------------
        # Holography panel (5)
        # ------------------------------

        holo_fft_button = QtWidgets.QPushButton('FFT', self)
        holo_sband_button = QtWidgets.QPushButton('Holo sband', self)
        holo_wref_sbands_button = QtWidgets.QPushButton('H+R sbands', self)
        holo_ifft_button = QtWidgets.QPushButton('IFFT', self)
        rec_holo_wref_auto_button = QtWidgets.QPushButton('Rec. Holo+Ref (Auto)', self)
        sum_button = QtWidgets.QPushButton('Sum', self)
        diff_button = QtWidgets.QPushButton('Diff', self)
        amplify_button = QtWidgets.QPushButton('Amplify', self)
        add_radians_button = QtWidgets.QPushButton('Add radians', self)
        remove_phase_tilt_button = QtWidgets.QPushButton('Remove phase tilt', self)
        get_sideband_from_xy_button = QtWidgets.QPushButton('Get sideband', self)

        self.subpixel_shift_checkbox = QtWidgets.QCheckBox('Subpixel shift (in dev.)', self)
        self.subpixel_shift_checkbox.setChecked(False)

        self.assign_rec_ph_to_obj_h_checkbox = QtWidgets.QCheckBox('(Rec. H+R) Assign reconstructed phase to object hologram', self)
        self.assign_rec_ph_to_obj_h_checkbox.setChecked(False)

        aperture_label = QtWidgets.QLabel('Aperture diam. [px]', self)
        self.aperture_input = QtWidgets.QLineEdit(str(const.aperture), self)
        self.aperture_input.setValidator(self.only_int)

        smooth_width_label = QtWidgets.QLabel('Smooth width [px]', self)
        self.smooth_width_input = QtWidgets.QLineEdit(str(const.smooth_width), self)
        self.smooth_width_input.setValidator(self.only_int)

        self.amp_factor_input = QtWidgets.QLineEdit('2.0', self)
        self.radians_to_add_input = QtWidgets.QLineEdit('3.14', self)

        sideband_xy_label = QtWidgets.QLabel('Sideband xy-coords:')
        self.sideband_x_input = QtWidgets.QLineEdit('0', self)
        self.sideband_y_input = QtWidgets.QLineEdit('0', self)

        holo_fft_button.clicked.connect(self.holo_fft)
        holo_sband_button.clicked.connect(self.holo_get_sideband)
        holo_wref_sbands_button.clicked.connect(self.holo_with_ref_get_sidebands)
        holo_ifft_button.clicked.connect(self.holo_ifft)
        rec_holo_wref_auto_button.clicked.connect(self.rec_holo_with_ref_auto)
        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)
        amplify_button.clicked.connect(self.amplify_phase)
        add_radians_button.clicked.connect(self.add_radians)
        remove_phase_tilt_button.clicked.connect(self.remove_phase_tilt)
        get_sideband_from_xy_button.clicked.connect(self.get_sideband_from_xy)

        self.tab_holo = QtWidgets.QWidget(self)
        self.tab_holo.layout = QtWidgets.QGridLayout()
        self.tab_holo.layout.setColumnStretch(0, 1)
        self.tab_holo.layout.setColumnStretch(1, 1)
        self.tab_holo.layout.setColumnStretch(2, 1)
        self.tab_holo.layout.setColumnStretch(3, 1)
        self.tab_holo.layout.setColumnStretch(4, 1)
        self.tab_holo.layout.setColumnStretch(5, 1)
        self.tab_holo.layout.setRowStretch(0, 1)
        self.tab_holo.layout.setRowStretch(8, 1)
        self.tab_holo.layout.addWidget(holo_fft_button, 1, 1)
        self.tab_holo.layout.addWidget(holo_sband_button, 1, 2)
        self.tab_holo.layout.addWidget(holo_ifft_button, 2, 1)
        self.tab_holo.layout.addWidget(holo_wref_sbands_button, 2, 2)
        self.tab_holo.layout.addWidget(rec_holo_wref_auto_button, 3, 1, 1, 2)
        self.tab_holo.layout.addWidget(self.subpixel_shift_checkbox, 4, 1, 1, 2)
        self.tab_holo.layout.addWidget(sum_button, 5, 1)
        self.tab_holo.layout.addWidget(diff_button, 5, 2)
        self.tab_holo.layout.addWidget(remove_phase_tilt_button, 6, 1, 1, 2)
        self.tab_holo.layout.addWidget(aperture_label, 1, 3)
        self.tab_holo.layout.addWidget(self.aperture_input, 1, 4)
        self.tab_holo.layout.addWidget(smooth_width_label, 2, 3)
        self.tab_holo.layout.addWidget(self.smooth_width_input, 2, 4)
        self.tab_holo.layout.addWidget(amplify_button, 3, 3)
        self.tab_holo.layout.addWidget(self.amp_factor_input, 3, 4)
        self.tab_holo.layout.addWidget(add_radians_button, 4, 3)
        self.tab_holo.layout.addWidget(self.radians_to_add_input, 4, 4)
        self.tab_holo.layout.addWidget(sideband_xy_label, 5, 3)
        self.tab_holo.layout.addWidget(self.sideband_x_input, 5, 4)
        self.tab_holo.layout.addWidget(self.sideband_y_input, 6, 4)
        self.tab_holo.layout.addWidget(get_sideband_from_xy_button, 6, 3)
        self.tab_holo.layout.addWidget(self.assign_rec_ph_to_obj_h_checkbox, 7, 1, 1, 4)
        self.tab_holo.setLayout(self.tab_holo.layout)

        # ------------------------------
        # Magnetic calculations panel (6)
        # ------------------------------

        plot_button = QtWidgets.QPushButton('Plot profile', self)
        calc_B_sec_button = QtWidgets.QPushButton('Calc. B from section', self)
        calc_B_prof_button = QtWidgets.QPushButton('Calc. B from profile')
        calc_grad_button = QtWidgets.QPushButton('Calculate gradient', self)
        calc_Bxy_maps_button = QtWidgets.QPushButton('Calc. Bx, By maps', self)
        calc_B_pol_button = QtWidgets.QPushButton('Calc. B polar', self)
        calc_B_pol_sectors_button = QtWidgets.QPushButton('Calc. B polar (n x m)', self)
        gen_phase_stats_button = QtWidgets.QPushButton('Gen. phase statistics', self)
        calc_MIP_button = QtWidgets.QPushButton('Calc. MIP', self)
        filter_contours_button = QtWidgets.QPushButton('Filter contours', self)

        self.orig_in_pt1_radio_button = QtWidgets.QRadioButton('Orig in pt1', self)
        self.orig_in_mid_radio_button = QtWidgets.QRadioButton('Orig in middle', self)
        self.orig_in_pt1_radio_button.setChecked(True)

        orig_B_pol_group = QtWidgets.QButtonGroup(self)
        orig_B_pol_group.addButton(self.orig_in_pt1_radio_button)
        orig_B_pol_group.addButton(self.orig_in_mid_radio_button)

        prof_width_label = QtWidgets.QLabel('Profile width [px]', self)
        self.prof_width_input = QtWidgets.QLineEdit('1', self)

        sample_thick_label = QtWidgets.QLabel('Sample thickness [nm]', self)
        self.sample_thick_input = QtWidgets.QLineEdit('30', self)

        threshold_label = QtWidgets.QLabel('Int. threshold [0-1]', self)
        self.threshold_input = QtWidgets.QLineEdit('0.9', self)

        num_of_r_iters_label = QtWidgets.QLabel('# R iters', self)
        self.num_of_r_iters_input = QtWidgets.QLineEdit('1', self)

        self.B_pol_n_rows_input = QtWidgets.QLineEdit('1', self)
        self.B_pol_n_cols_input = QtWidgets.QLineEdit('1', self)

        acc_voltage_label = QtWidgets.QLabel('U_acc [kV]', self)
        self.acc_voltage_input = QtWidgets.QLineEdit('300', self)

        plot_button.clicked.connect(self.plot_profile)
        calc_B_sec_button.clicked.connect(self.calc_B_from_section)
        calc_B_prof_button.clicked.connect(self.calc_B_from_profile)
        calc_grad_button.clicked.connect(self.calc_phase_gradient)
        calc_Bxy_maps_button.clicked.connect(self.calc_Bxy_maps)
        calc_B_pol_button.clicked.connect(self.calc_B_polar_from_section)
        calc_B_pol_sectors_button.clicked.connect(partial(self.calc_B_polar_from_section, True))
        gen_phase_stats_button.clicked.connect(self.gen_phase_stats)
        calc_MIP_button.clicked.connect(self.calc_mean_inner_potential)
        filter_contours_button.clicked.connect(self.filter_contours)

        self.tab_calc = QtWidgets.QWidget(self)
        self.tab_calc.layout = QtWidgets.QGridLayout()
        self.tab_calc.layout.setColumnStretch(0, 1)
        self.tab_calc.layout.setColumnStretch(1, 1)
        self.tab_calc.layout.setColumnStretch(2, 1)
        self.tab_calc.layout.setColumnStretch(3, 1)
        self.tab_calc.layout.setColumnStretch(4, 1)
        self.tab_calc.layout.setColumnStretch(5, 1)
        self.tab_calc.layout.setColumnStretch(6, 1)
        self.tab_calc.layout.setColumnStretch(7, 1)
        self.tab_calc.layout.setRowStretch(0, 1)
        self.tab_calc.layout.setRowStretch(8, 1)
        self.tab_calc.layout.addWidget(sample_thick_label, 1, 1, 1, 2)
        self.tab_calc.layout.addWidget(self.sample_thick_input, 2, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_grad_button, 3, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_sec_button, 4, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_prof_button, 5, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_Bxy_maps_button, 6, 1, 1, 2)
        self.tab_calc.layout.addWidget(gen_phase_stats_button, 7, 1, 1, 2)
        self.tab_calc.layout.addWidget(prof_width_label, 1, 3, 1, 2)
        self.tab_calc.layout.addWidget(self.prof_width_input, 2, 3, 1, 2)
        self.tab_calc.layout.addWidget(plot_button, 3, 3, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_pol_button, 4, 3, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_pol_sectors_button, 5, 3, 1, 2)
        self.tab_calc.layout.addWidget(self.orig_in_pt1_radio_button, 6, 3, 1, 2)
        self.tab_calc.layout.addWidget(self.orig_in_mid_radio_button, 7, 3, 1, 2)
        self.tab_calc.layout.addWidget(threshold_label, 1, 5, 1, 2)
        self.tab_calc.layout.addWidget(self.threshold_input, 2, 5, 1, 2)
        self.tab_calc.layout.addWidget(filter_contours_button, 3, 5, 1, 2)
        self.tab_calc.layout.addWidget(num_of_r_iters_label, 4, 5)
        self.tab_calc.layout.addWidget(self.num_of_r_iters_input, 4, 6)
        self.tab_calc.layout.addWidget(self.B_pol_n_rows_input, 5, 5)
        self.tab_calc.layout.addWidget(self.B_pol_n_cols_input, 5, 6)
        self.tab_calc.layout.addWidget(acc_voltage_label, 6, 5)
        self.tab_calc.layout.addWidget(self.acc_voltage_input, 6, 6)
        self.tab_calc.layout.addWidget(calc_MIP_button, 7, 5, 1, 2)
        self.tab_calc.setLayout(self.tab_calc.layout)

        # ------------------------------
        # Magnetic calculations panel #2 (7)
        # ------------------------------

        export_glob_scaled_phases_button = QtWidgets.QPushButton('Export phase colormaps', self)
        export_3d_phase_button = QtWidgets.QPushButton('Export 3D phase', self)

        self.add_arrows_checkbox = QtWidgets.QCheckBox('Add grad. arrows', self)
        self.add_arrows_checkbox.setChecked(False)

        self.perpendicular_arrows_checkbox = QtWidgets.QCheckBox('Perpendicular', self)
        self.perpendicular_arrows_checkbox.setChecked(False)

        arr_size_label = QtWidgets.QLabel('Arrow size [au]', self)
        arr_dist_label = QtWidgets.QLabel('Arrow dist. [px]', self)
        self.arr_size_input = QtWidgets.QLineEdit('20', self)
        self.arr_dist_input = QtWidgets.QLineEdit('50', self)

        self.arr_size_input.setValidator(self.only_int)
        self.arr_dist_input.setValidator(self.only_int)

        ph3d_elev_label = QtWidgets.QLabel('Elev. angle [{0}]'.format(u'\N{DEGREE SIGN}'), self)
        ph3d_azim_label = QtWidgets.QLabel('Azim. angle [{0}]'.format(u'\N{DEGREE SIGN}'), self)
        self.ph3d_elev_input = QtWidgets.QLineEdit('0', self)
        self.ph3d_azim_input = QtWidgets.QLineEdit('0', self)

        ph3d_mesh_label = QtWidgets.QLabel('Mesh dist. [px]', self)
        self.ph3d_mesh_input = QtWidgets.QLineEdit('50', self)
        self.ph3d_mesh_input.setValidator(self.only_int)

        export_glob_scaled_phases_button.clicked.connect(self.export_glob_sc_phases)
        export_3d_phase_button.clicked.connect(self.export_3d_phase)

        self.tab_calc_2 = QtWidgets.QWidget(self)
        self.tab_calc_2.layout = QtWidgets.QGridLayout()
        self.tab_calc_2.layout.setColumnStretch(0, 1)
        self.tab_calc_2.layout.setColumnStretch(1, 1)
        self.tab_calc_2.layout.setColumnStretch(2, 1)
        self.tab_calc_2.layout.setColumnStretch(3, 1)
        self.tab_calc_2.layout.setColumnStretch(4, 1)
        self.tab_calc_2.layout.setColumnStretch(5, 1)
        self.tab_calc_2.layout.setRowStretch(0, 1)
        self.tab_calc_2.layout.setRowStretch(6, 1)
        self.tab_calc_2.layout.addWidget(arr_size_label, 1, 1)
        self.tab_calc_2.layout.addWidget(self.arr_size_input, 1, 2)
        self.tab_calc_2.layout.addWidget(arr_dist_label, 2, 1)
        self.tab_calc_2.layout.addWidget(self.arr_dist_input, 2, 2)
        self.tab_calc_2.layout.addWidget(export_glob_scaled_phases_button, 3, 1, 1, 2)
        self.tab_calc_2.layout.addWidget(self.add_arrows_checkbox, 4, 1, 1, 2)
        self.tab_calc_2.layout.addWidget(self.perpendicular_arrows_checkbox, 5, 1, 1, 2)
        self.tab_calc_2.layout.addWidget(ph3d_elev_label, 1, 3)
        self.tab_calc_2.layout.addWidget(self.ph3d_elev_input, 1, 4)
        self.tab_calc_2.layout.addWidget(ph3d_azim_label, 2, 3)
        self.tab_calc_2.layout.addWidget(self.ph3d_azim_input, 2, 4)
        self.tab_calc_2.layout.addWidget(ph3d_mesh_label, 3, 3)
        self.tab_calc_2.layout.addWidget(self.ph3d_mesh_input, 3, 4)
        self.tab_calc_2.layout.addWidget(export_3d_phase_button, 4, 3, 1, 2)
        self.tab_calc_2.setLayout(self.tab_calc_2.layout)

        # ------------------------------
        # Bright/Gamma/Contrast panel (8)
        # ------------------------------

        reset_bright_button = QtWidgets.QPushButton('Reset B', self)
        reset_cont_button = QtWidgets.QPushButton('Reset C', self)
        reset_gamma_button = QtWidgets.QPushButton('Reset G', self)

        bright_label = QtWidgets.QLabel('Brightness', self)
        cont_label = QtWidgets.QLabel('Contrast', self)
        gamma_label = QtWidgets.QLabel('Gamma', self)

        self.bright_input = QtWidgets.QLineEdit('0', self)
        self.cont_input = QtWidgets.QLineEdit('255', self)
        self.gamma_input = QtWidgets.QLineEdit('1.0', self)

        self.bright_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright_slider.setFixedHeight(14)
        self.bright_slider.setRange(-255, 255)
        self.bright_slider.setValue(0)

        self.cont_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cont_slider.setFixedHeight(14)
        self.cont_slider.setRange(1, 1785)
        self.cont_slider.setValue(255)

        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setFixedHeight(14)
        self.gamma_slider.setRange(10, 190)
        self.gamma_slider.setValue(100)

        reset_bright_button.clicked.connect(self.reset_bright)
        reset_cont_button.clicked.connect(self.reset_cont)
        reset_gamma_button.clicked.connect(self.reset_gamma)

        self.bright_slider.valueChanged.connect(self.disp_bright_value)
        self.cont_slider.valueChanged.connect(self.disp_cont_value)
        self.gamma_slider.valueChanged.connect(self.disp_gamma_value)

        self.bright_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.cont_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.gamma_slider.sliderReleased.connect(self.update_display_and_bcg)

        self.bright_input.returnPressed.connect(self.update_display_and_bcg)
        self.cont_input.returnPressed.connect(self.update_display_and_bcg)
        self.gamma_input.returnPressed.connect(self.update_display_and_bcg)

        self.tab_corr = QtWidgets.QWidget(self)
        self.tab_corr.layout = QtWidgets.QGridLayout()
        self.tab_corr.layout.setColumnStretch(0, 1)
        self.tab_corr.layout.setColumnStretch(1, 2)
        self.tab_corr.layout.setColumnStretch(2, 1)
        self.tab_corr.layout.setColumnStretch(3, 1)
        self.tab_corr.layout.setColumnStretch(4, 1)
        self.tab_corr.layout.setRowStretch(0, 1)
        self.tab_corr.layout.setRowStretch(7, 1)
        self.tab_corr.layout.addWidget(bright_label, 1, 2)
        self.tab_corr.layout.addWidget(self.bright_slider, 2, 1)
        self.tab_corr.layout.addWidget(self.bright_input, 2, 2)
        self.tab_corr.layout.addWidget(reset_bright_button, 2, 3)
        self.tab_corr.layout.addWidget(cont_label, 3, 2)
        self.tab_corr.layout.addWidget(self.cont_slider, 4, 1)
        self.tab_corr.layout.addWidget(self.cont_input, 4, 2)
        self.tab_corr.layout.addWidget(reset_cont_button, 4, 3)
        self.tab_corr.layout.addWidget(gamma_label, 5, 2)
        self.tab_corr.layout.addWidget(self.gamma_slider, 6, 1)
        self.tab_corr.layout.addWidget(self.gamma_input, 6, 2)
        self.tab_corr.layout.addWidget(reset_gamma_button, 6, 3)
        self.tab_corr.setLayout(self.tab_corr.layout)

        # ------------------------------
        # Main layout
        # ------------------------------

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.addTab(self.tab_nav, 'Navigation')
        self.tabs.addTab(self.tab_disp, 'Display')
        self.tabs.addTab(self.tab_align, 'Alignment')
        self.tabs.addTab(self.tab_holo, 'Holography')
        self.tabs.addTab(self.tab_calc, 'Mag. field #1')
        self.tabs.addTab(self.tab_calc_2, 'Mag. field #2')
        self.tabs.addTab(self.tab_corr, 'Corrections')

        self.disable_tabs()
        self.tab_nav.setEnabled(True)

        vbox_panel = QtWidgets.QVBoxLayout()
        vbox_panel.addWidget(self.curr_info_label)
        vbox_panel.addWidget(self.tabs)
        vbox_panel.addWidget(self.plot_widget)
        # vbox_panel.addWidget(self.preview_scroll)

        hbox_main = QtWidgets.QHBoxLayout()
        hbox_main.addWidget(self.display)
        hbox_main.addLayout(vbox_panel)

        self.setLayout(hbox_main)

    def update_curr_info_label(self):
        if self.display.image is None:
            return
        curr_img = self.display.image
        disp_name = curr_img.name[:const.disp_name_max_len]
        if len(curr_img.name) > const.disp_name_max_len:
            disp_name = disp_name[:-3] + '...'
        self.curr_info_label.setText('{0}: {1}, dim = {2} px'.format(curr_img.num_in_ser, disp_name, curr_img.width))

    def enable_tabs(self):
        # self.tab_nav.setEnabled(True)
        self.tab_disp.setEnabled(True)
        self.tab_align.setEnabled(True)
        self.tab_holo.setEnabled(True)
        self.tab_calc.setEnabled(True)
        self.tab_calc_2.setEnabled(True)
        self.tab_corr.setEnabled(True)

    def disable_tabs(self):
        # self.tab_nav.setEnabled(False)
        self.tab_disp.setEnabled(False)
        self.tab_align.setEnabled(False)
        self.tab_holo.setEnabled(False)
        self.tab_calc.setEnabled(False)
        self.tab_calc_2.setEnabled(False)
        self.tab_corr.setEnabled(False)

    def enable_manual_panel(self):
        self.move_left_button.setEnabled(True)
        self.move_right_button.setEnabled(True)
        self.move_up_button.setEnabled(True)
        self.move_down_button.setEnabled(True)
        self.rot_clockwise_button.setEnabled(True)
        self.rot_counter_clockwise_button.setEnabled(True)
        self.px_shift_input.setEnabled(True)
        self.rot_angle_input.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    def disable_manual_panel(self):
        if self.backup_image is not None:
            self.reset_changes_and_delete_backup(upd_disp=False)
        self.move_left_button.setEnabled(False)
        self.move_right_button.setEnabled(False)
        self.move_up_button.setEnabled(False)
        self.move_down_button.setEnabled(False)
        self.rot_clockwise_button.setEnabled(False)
        self.rot_counter_clockwise_button.setEnabled(False)
        self.px_shift_input.setEnabled(False)
        self.rot_angle_input.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.reset_button.setEnabled(False)

    def set_image_name(self):
        self.display.image.name = self.name_input.text()
        self.fname_input.setText(self.name_input.text())
        self.update_curr_info_label()

    def reset_image_names(self):
        curr_img = self.display.image
        first_img = imsup.get_first_image(curr_img)
        img_queue = imsup.create_image_list_from_first_image(first_img)
        for img, idx in zip(img_queue, range(len(img_queue))):
            img.name = 'img_0{0}'.format(idx + 1) if idx < 9 else 'img_{0}'.format(idx + 1)
        self.update_curr_info_label()
        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)

    def go_to_image(self, new_idx):
        first_img = imsup.get_first_image(self.display.image)
        imgs = imsup.create_image_list_from_first_image(first_img)
        if 0 > new_idx >= len(imgs):
            print('Image index out of range')
            return

        curr_img = imgs[new_idx]
        if curr_img.name == '':
            curr_img.name = 'img_0{0}'.format(new_idx + 1) if new_idx < 9 else 'img_{0}'.format(new_idx + 1)

        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)
        self.manual_mode_checkbox.setChecked(False)
        self.disable_manual_panel()
        self.display.image = imgs[new_idx]
        self.display.update_labels()
        self.update_curr_info_label()
        self.update_display_and_bcg()

    def go_to_prev_image(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            return
        prev_idx = curr_img.prev.num_in_ser - 1
        self.go_to_image(prev_idx)

    def go_to_next_image(self):
        curr_img = self.display.image
        if curr_img.next is None:
            return
        next_idx = curr_img.next.num_in_ser - 1
        self.go_to_image(next_idx)

    def go_to_first_image(self):
        curr_idx = self.display.image.num_in_ser - 1
        if curr_idx > 0:
            self.go_to_image(0)

    def go_to_last_image(self):
        curr_img = self.display.image
        last_img = imsup.get_last_image(curr_img)
        curr_idx = curr_img.num_in_ser - 1
        last_idx = last_img.num_in_ser - 1
        if curr_idx < last_idx:
            self.go_to_image(last_idx)

    def insert_img_after_curr(self, new_img):
        curr_num = self.display.image.num_in_ser
        curr_img_list = imsup.create_image_list_from_first_image(self.display.image)
        new_img_list = imsup.create_image_list_from_first_image(new_img)

        curr_img_list[1:1] = new_img_list
        self.point_sets[abs(curr_num):abs(curr_num)] = [[] for _ in range(len(new_img_list))]

        curr_img_list.update_links()
        self.go_to_image(abs(curr_num))

        if curr_num == -1:  # starting (blank) image is identified by specific number (-1)
            del curr_img_list[0]
            del self.point_sets[0]
            self.update_curr_info_label()

        # self.preview_scroll.update_scroll_list(self.display.image)

    def move_image_to_index(self, new_idx):
        curr_img = self.display.image
        curr_idx = abs(curr_img.num_in_ser) - 1

        if new_idx == curr_idx or new_idx < 0: return

        first_img = imsup.get_first_image(curr_img)
        imgs = imsup.create_image_list_from_first_image(first_img)

        if new_idx >= len(imgs): return

        imgs.insert(new_idx, imgs.pop(curr_idx))
        self.point_sets.insert(new_idx, self.point_sets.pop(curr_idx))
        imgs.update_and_restrain_links()
        self.go_to_image(new_idx)

    def swap_left(self):
        prev_idx = self.display.image.num_in_ser - 2
        self.move_image_to_index(prev_idx)

    def swap_right(self):
        next_idx = self.display.image.num_in_ser
        self.move_image_to_index(next_idx)

    def make_image_first(self):
        self.move_image_to_index(0)

    def make_image_last(self):
        last_img = imsup.get_last_image(self.display.image)
        self.move_image_to_index(last_img.num_in_ser - 1)

    def clear_image(self):
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
        curr_idx = abs(self.display.image.num_in_ser) - 1
        self.point_sets[curr_idx][:] = []
        self.display.repaint()

    def remove_last_marker(self):
        curr_idx = abs(self.display.image.num_in_ser) - 1
        if len(self.point_sets[curr_idx]) == 0:
            return
        all_labels = self.display.children()
        if len(all_labels) > 0:
            last_label = all_labels[-1]
            last_label.deleteLater()
        del self.point_sets[curr_idx][-1]
        self.display.repaint()

    def add_marker_at_xy(self):
        curr_img = self.display.image
        curr_idx = abs(curr_img.num_in_ser) - 1
        curr_pos = [int(self.marker_x_input.text()), int(self.marker_y_input.text())]
        if 0 <= curr_pos[0] < const.disp_dim and 0 <= curr_pos[1] < const.disp_dim:
            # --- to be removed later ---
            # for idx in range(len(self.point_sets)):
            #     self.point_sets[idx].append(curr_pos)
            # ---------------------------
            self.point_sets[curr_idx].append(curr_pos)  # uncomment later
            self.display.repaint()
            if self.display.show_labs:
                self.display.show_last_label()

            pt_idx = len(self.point_sets[curr_idx])
            disp_x, disp_y = curr_pos
            real_x, real_y = disp_pt_to_real_tl_pt(curr_img.width, curr_pos)
            print('Added point {0} at:\nx = {1}\ny = {2}'.format(pt_idx, disp_x, disp_y))
            print('Actual position:\nx = {0}\ny = {1}'.format(real_x, real_y))
            print('Amp = {0:.2f}\nPhs = {1:.2f}'.format(curr_img.amph.am[real_y, real_x],
                                                        curr_img.amph.ph[real_y, real_x]))
        else:
            print('Wrong marker coordinates')

    def delete_image(self):
        curr_img = self.display.image
        if curr_img.prev is None and curr_img.next is None:
            return

        curr_idx = curr_img.num_in_ser - 1
        first_img = imsup.get_first_image(curr_img)
        all_img_list = imsup.create_image_list_from_first_image(first_img)

        new_idx = curr_idx - 1 if curr_img.prev is not None else curr_idx + 1
        self.go_to_image(new_idx)

        del all_img_list[curr_idx]      # ImageList destructor updates and restrains links among the remaining images
        del self.point_sets[curr_idx]
        self.update_curr_info_label()

    def transfer_phase_to_image(self):
        curr_img = self.display.image
        curr_idx = abs(curr_img.num_in_ser) - 1
        to_idx = int(self.img_num_for_phs_transfer_input.text()) - 1

        if to_idx == curr_idx:
            print('Choose another image')
            return

        first_img = imsup.get_first_image(curr_img)
        imgs = imsup.create_image_list_from_first_image(first_img)

        if to_idx < 0 or to_idx >= len(imgs):
            print('Image index out of range')
            return

        if imgs[to_idx].amph.ph.shape != curr_img.amph.ph.shape:
            print('Transfer not possible: images have different sizes')
            return

        imgs[to_idx].amph.ph = np.copy(curr_img.amph.ph)
        imgs[to_idx] = rescale_image_buffer_to_window(imgs[to_idx], const.disp_dim)
        self.go_to_image(to_idx)
        print('Phase transferred: {0} --> {1}'.format(curr_idx + 1, to_idx + 1))

    def toggle_lines(self):
        self.display.show_lines = not self.display.show_lines
        self.display.repaint()

    def toggle_labels(self):
        self.display.show_labs = not self.display.show_labs
        if self.display.show_labs:
            self.display.show_labels()
        else:
            self.display.hide_labels()

    def toggle_log_scale(self):
        self.log_scale_checkbox.setChecked(not self.log_scale_checkbox.isChecked())
        self.update_display_and_bcg()

    def toggle_hide_bad_pixels(self):
        self.hide_bad_px_checkbox.setChecked(not self.hide_bad_px_checkbox.isChecked())
        self.update_display_and_bcg()

    def update_display(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_hide_bad_px_checked = self.hide_bad_px_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.set_image(disp_amp=is_amp_checked, disp_phs=is_phs_checked, log_scale=is_log_scale_checked,
                               hide_bad_px=is_hide_bad_px_checked, color=is_color_checked)

    def update_bcg(self):
        bright_val = int(self.bright_input.text())
        cont_val = int(self.cont_input.text())
        gamma_val = float(self.gamma_input.text())

        self.change_bright_slider_value()
        self.change_cont_slider_value()
        self.change_gamma_slider_value()

        self.display.set_image(update_bcg=True, bright=bright_val, cont=cont_val, gamma=gamma_val)

    def update_display_and_bcg(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_hide_bad_px_checked = self.hide_bad_px_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()

        bright_val = int(self.bright_input.text())
        cont_val = int(self.cont_input.text())
        gamma_val = float(self.gamma_input.text())

        self.change_bright_slider_value()
        self.change_cont_slider_value()
        self.change_gamma_slider_value()

        self.display.set_image(disp_amp=is_amp_checked, disp_phs=is_phs_checked, log_scale=is_log_scale_checked,
                               hide_bad_px=is_hide_bad_px_checked, color=is_color_checked,
                               update_bcg=True, bright=bright_val, cont=cont_val, gamma=gamma_val)

    def disp_bright_value(self):
        self.bright_input.setText('{0:.0f}'.format(self.bright_slider.value()))

    def disp_cont_value(self):
        self.cont_input.setText('{0:.0f}'.format(self.cont_slider.value()))

    def disp_gamma_value(self):
        self.gamma_input.setText('{0:.2f}'.format(self.gamma_slider.value() * 0.01))

    def change_bright_slider_value(self):
        b = int(self.bright_input.text())
        b_min = self.bright_slider.minimum()
        b_max = self.bright_slider.maximum()
        if b < b_min:
            b = b_min
        elif b > b_max:
            b = b_max
        self.bright_slider.setValue(b)

    def change_cont_slider_value(self):
        c = int(self.cont_input.text())
        c_min = self.cont_slider.minimum()
        c_max = self.cont_slider.maximum()
        if c < c_min:
            c = c_min
        elif c > c_max:
            c = c_max
        self.cont_slider.setValue(c)

    def change_gamma_slider_value(self):
        g = float(self.gamma_input.text()) * 100
        g_min = self.gamma_slider.minimum()
        g_max = self.gamma_slider.maximum()
        if g < g_min:
            g = g_min
        elif g > g_max:
            g = g_max
        self.gamma_slider.setValue(g)

    def reset_bright(self):
        self.bright_input.setText('0')
        self.update_display_and_bcg()

    def reset_cont(self):
        self.cont_input.setText('255')
        self.update_display_and_bcg()

    def reset_gamma(self):
        self.gamma_input.setText('1.0')
        self.update_display_and_bcg()

    def unwrap_phase(self):
        curr_img = self.display.image
        new_phs = tr.unwrap_phase(curr_img.amph.ph)
        curr_img.amph.ph = np.copy(new_phs)
        self.display.image = rescale_image_buffer_to_window(curr_img, const.disp_dim)
        self.update_display()

    def wrap_phase(self):
        curr_img = self.display.image
        uw_min = np.min(curr_img.amph.ph)

        if uw_min > 0:
            uw_min = 0
        new_phs = (curr_img.amph.ph - uw_min) % (2 * np.pi) - np.pi

        curr_img.amph.ph = np.copy(new_phs)
        self.display.image = rescale_image_buffer_to_window(curr_img, const.disp_dim)
        self.update_display()

    def blank_area(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1

        if len(self.point_sets[curr_idx]) < 2:
            print('Mark two points to indicate the area...')
            return

        p1, p2 = self.point_sets[curr_idx][:2]
        p1, p2 = tr.convert_points_to_tl_br(p1, p2)
        p1 = disp_pt_to_real_tl_pt(curr_img.width, p1)
        p2 = disp_pt_to_real_tl_pt(curr_img.width, p2)

        blanked_img = imsup.copy_amph_image(curr_img)
        blanked_img.amph.am[p1[1]:p2[1], p1[0]:p2[0]] = 0.0
        blanked_img.amph.ph[p1[1]:p2[1], p1[0]:p2[0]] = 0.0

        blanked_img.name = '{0}_b'.format(curr_img.name)
        self.insert_img_after_curr(blanked_img)

    def norm_phase(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1

        n_points = len(self.point_sets[curr_idx])
        if n_points == 0:
            print('Mark reference point (or area -- two points) on the image')
            return

        pt_disp = self.point_sets[curr_idx][:2]
        pt_real = disp_pts_to_real_tl_pts(curr_img.width, pt_disp)

        if n_points == 1:
            x1, y1 = pt_real[0]
            x2, y2 = 0, 0
        else:
            (x1, y1), (x2, y2) = tr.convert_points_to_tl_br(pt_real[0], pt_real[1])

        first_img = imsup.get_first_image(curr_img)
        img_list = imsup.create_image_list_from_first_image(first_img)

        for img in img_list:
            if n_points == 1:
                norm_val = img.amph.ph[y1, x1]
            else:
                norm_val = np.average(img.amph.ph[y1:y2, x1:x2])
            img.amph.ph -= norm_val

            if img.cos_phase is not None:
                img.update_cos_phase()

            img.name += '_-' if norm_val > 0.0 else '_+'
            img.name += '{0:.2f}rad'.format(abs(norm_val))

        self.fname_input.setText(curr_img.name)
        self.update_display_and_bcg()
        self.update_curr_info_label()
        print('All phases normalized')

    def flip_image(self, horizontal=True):
        curr_img = self.display.image
        if horizontal:
            imsup.flip_image_h(curr_img)
        else:
            imsup.flip_image_v(curr_img)
        self.display.image = rescale_image_buffer_to_window(curr_img, const.disp_dim)  # refresh buffer
        self.update_display_and_bcg()

    def export_image(self):
        curr_num = self.display.image.num_in_ser
        curr_img = self.display.image
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        fname = self.fname_input.text()
        if fname == '':
            if is_amp_checked:
                fname = 'amp{0}'.format(curr_num)
            elif is_phs_checked:
                fname = 'phs{0}'.format(curr_num)
            else:
                fname = 'cos_phs{0}'.format(curr_num)

        if is_amp_checked:
            img_type = 'amplitude'
            data_to_export = np.copy(curr_img.amph.am)
        elif is_phs_checked:
            img_type = 'phase'
            data_to_export = np.copy(curr_img.amph.ph)
        else:
            img_type = 'cos(phase)'
            data_to_export = np.cos(curr_img.amph.ph)

        # numpy array file (new)
        if self.export_npy_radio_button.isChecked():
            fname_ext = '.npy'
            np.save(fname, data_to_export)
            print('Saved image to numpy array file: "{0}.npy"'.format(fname))

        # raw data file
        elif self.export_raw_radio_button.isChecked():
            fname_ext = ''
            data_to_export.tofile(fname)
            print('Saved image to raw data file: "{0}"'.format(fname))

        # TIF file
        else:
            fname_ext = '.tif'
            log_scale = True if is_amp_checked and self.log_scale_checkbox.isChecked() else False
            hide_bad_px = True if is_amp_checked and self.hide_bad_px_checkbox.isChecked() else False
            color = True if self.color_radio_button.isChecked() else False
            bright = int(self.bright_input.text())
            cont = int(self.cont_input.text())
            gamma = float(self.gamma_input.text())

            imsup.save_arr_as_tiff(data_to_export, fname, log_scale, color, bright, cont, gamma,
                                   hide_bad_px, const.min_px_threshold, const.max_px_threshold)

            print('Saved image as "{0}.tif"'.format(fname))
            # if log_scale: print('Warning: Logarithmic scale is on')
            # if hide_bad_px: print('Warning: Outlier pixels are hidden')

        # save log file
        log_fname = '{0}_log.txt'.format(fname)
        with open(log_fname, 'w') as log_file:
            log_file.write('File name:\t{0}{1}\n'
                           'Image name:\t{2}\n'
                           'Image size:\t{3}x{4}\n'
                           'Image type:\t{5}\n'
                           'Data type:\t{6}\n'
                           'Calibration:\t{7} nm\n'.format(fname, fname_ext, curr_img.name, curr_img.width,
                                                           curr_img.height, img_type, data_to_export.dtype,
                                                           curr_img.px_dim * 1e9))
        print('Saved log file: "{0}"'.format(log_fname))

    def export_all(self):
        curr_img = imsup.get_first_image(self.display.image)
        self.display.image = curr_img
        self.export_image()
        while curr_img.next is not None:
            self.go_to_next_image()
            self.export_image()
            curr_img = self.display.image
        print('All images saved')

    def export_3d_phase(self):
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

        curr_img = self.display.image
        img_dim = curr_img.width
        px_sz = curr_img.px_dim * 1e9

        elev_ang = int(self.ph3d_elev_input.text())
        azim_ang = int(self.ph3d_azim_input.text())
        step = int(self.ph3d_mesh_input.text())

        # X = np.arange(0, img_dim, step, dtype=np.float32)
        # Y = np.arange(0, img_dim, step, dtype=np.float32)
        # phs_to_disp = np.copy(curr_img.amph.ph[0:img_dim:step, 0:img_dim:step])

        X = np.arange(0, img_dim, dtype=np.float32)
        Y = np.arange(0, img_dim, dtype=np.float32)
        X, Y = np.meshgrid(X, Y)
        X *= px_sz
        Y *= px_sz

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, curr_img.amph.ph, cmap=cm.jet, rstride=step, cstride=step)    # mesh step (dist. between rows/cols used)
        # ax.plot_surface(X, Y, curr_img.amph.ph, cmap=cm.jet, rcount=step, ccount=step)    # mesh (how many rows, cols will be used)
        # ax.plot_surface(X, Y, phs_to_disp, cmap=cm.jet)

        ax.view_init(elev_ang, azim_ang)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)       # reduce white spaces around 3d plot
        fig.savefig('{0}_{1}_{2}.png'.format(curr_img.name, elev_ang, azim_ang), dpi=300)
        ax.cla()
        fig.clf()
        plt.close(fig)
        print('3D phase image exported!')

    def export_glob_sc_phases(self):
        first_img = imsup.get_first_image(self.display.image)
        img_list = imsup.create_image_list_from_first_image(first_img)
        is_arrows_checked = self.add_arrows_checkbox.isChecked()
        is_perpendicular_checked = self.perpendicular_arrows_checkbox.isChecked()
        arrow_size = int(self.arr_size_input.text())
        arrow_dist = int(self.arr_dist_input.text())
        export_glob_sc_images(img_list, is_arrows_checked, is_perpendicular_checked, arrow_size, arrow_dist, cbar_lab='phase shift [rad]')
        print('Phases exported!')

    def crop_n_fragments(self):
        curr_idx = self.display.image.num_in_ser - 1
        if len(self.point_sets[curr_idx]) < 2:
            print('Mark two points to indicate the cropping area...')
            return

        curr_img = self.display.image
        pt1, pt2 = self.point_sets[curr_idx][:2]
        pt1, pt2 = tr.convert_points_to_tl_br(pt1, pt2)
        disp_crop_coords = pt1 + pt2
        real_tl_coords = disp_pt_to_real_tl_pt(curr_img.width, disp_crop_coords)
        real_sq_coords = imsup.make_square_coords(real_tl_coords)
        if np.abs(real_sq_coords[2] - real_sq_coords[0]) % 2:
            real_sq_coords[2] += 1
            real_sq_coords[3] += 1

        print('ROI coords.: {0}'.format(real_sq_coords))

        n_to_crop = np.int(self.n_to_crop_input.text())
        first_img = imsup.get_first_image(curr_img)
        insert_idx = curr_idx + n_to_crop
        img_list = imsup.create_image_list_from_first_image(first_img)
        img_list2 = img_list[curr_idx:insert_idx]

        for img, n in zip(img_list2, range(insert_idx, insert_idx + n_to_crop)):
            frag = crop_fragment(img, real_sq_coords)
            frag.name = 'crop_from_{0}'.format(img.name)
            print('New dimensions: ({0}, {1}) px'.format(frag.width, frag.height))
            img_list.insert(n, frag)
            self.point_sets.insert(n, [])

        img_list.update_links()

        if self.clear_prev_checkbox.isChecked():
            del img_list[curr_idx:insert_idx]
            del self.point_sets[curr_idx:insert_idx]

        self.go_to_image(curr_idx)
        print('Cropping complete!')

    def create_backup_image(self):
        if self.manual_mode_checkbox.isChecked():
            self.backup_image = imsup.copy_amph_image(self.display.image)
            self.enable_manual_panel()
        else:
            self.reset_changes_and_delete_backup(upd_disp=True)
            self.disable_manual_panel()

    def move_left(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([-n_px, 0])

    def move_right(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([n_px, 0])

    def move_up(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([0, -n_px])

    def move_down(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([0, n_px])

    def move_image(self, shift):
        bckp = self.backup_image
        curr = self.display.image
        total_shift = list(np.array(curr.shift) + np.array(shift))

        if curr.rot != 0:
            tmp = tr.rotate_image_ski(bckp, curr.rot)
            shifted_img = imsup.shift_amph_image(tmp, total_shift)
        else:
            shifted_img = imsup.shift_amph_image(bckp, total_shift)

        curr.amph.am = np.copy(shifted_img.amph.am)
        curr.amph.ph = np.copy(shifted_img.amph.ph)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        self.display.image.shift = total_shift
        self.update_display_and_bcg()

    def rotate_left(self):
        ang = float(self.rot_angle_input.text())
        self.rotate_image(ang)

    def rotate_right(self):
        ang = float(self.rot_angle_input.text())
        self.rotate_image(-ang)

    def rotate_image(self, rot):
        bckp = self.backup_image
        curr = self.display.image
        total_rot = curr.rot + rot

        if curr.shift != [0, 0]:
            tmp = imsup.shift_amph_image(bckp, curr.shift)
            rotated_img = tr.rotate_image_ski(tmp, total_rot)
        else:
            rotated_img = tr.rotate_image_ski(bckp, total_rot)

        curr.amph.am = np.copy(rotated_img.amph.am)
        curr.amph.ph = np.copy(rotated_img.amph.ph)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        self.display.image.rot = total_rot
        self.update_display_and_bcg()

    def zero_shift_rot(self):
        self.display.image.shift = [0, 0]
        self.display.image.rot = 0

    def apply_changes(self):
        self.zero_shift_rot()
        self.backup_image = imsup.copy_amph_image(self.display.image)
        print('Changes to {0} have been applied'.format(self.display.image.name))

    def reset_changes(self, upd_disp=True):
        curr = self.display.image
        curr.amph.am = np.copy(self.backup_image.amph.am)
        curr.amph.ph = np.copy(self.backup_image.amph.ph)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        if curr.shift != [0, 0] or curr.rot != 0:
            if upd_disp: self.update_display_and_bcg()
            self.zero_shift_rot()
            print('Changes to {0} have been revoked'.format(self.display.image.name))

    def reset_changes_and_delete_backup(self, upd_disp=True):
        self.reset_changes(upd_disp)
        self.backup_image = None

    def cross_corr_with_prev(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            print('Could not find the reference (preceding) image...')
            return
        img_list_to_cc = imsup.create_image_list_from_image(curr_img.prev, 2)
        img_aligned = cross_corr_images(img_list_to_cc)[0]
        self.insert_img_after_curr(img_aligned)
        self.go_to_next_image()

    def cross_corr_all(self):
        curr_img = self.display.image
        first_img = imsup.get_first_image(curr_img)
        all_img_list = imsup.create_image_list_from_first_image(first_img)
        n_imgs = len(all_img_list)
        insert_idx = n_imgs
        img_align_list = cross_corr_images(all_img_list)

        ref_img = imsup.copy_amph_image(first_img)
        img_align_list.insert(0, ref_img)
        all_img_list += img_align_list
        for i in range(n_imgs):
            self.point_sets.append([])
        all_img_list.update_and_restrain_links()

        self.go_to_image(insert_idx)
        print('Cross-correlation done!')

    def auto_shift_image(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1
        img_width = curr_img.width

        if curr_img.prev is None:
            print('Could not find the reference (preceding) image...')
            return

        points1 = self.point_sets[curr_idx - 1]
        points2 = self.point_sets[curr_idx]

        if not validate_two_point_sets(points1, points2, min_sz=1):
            return

        set1 = disp_pts_to_real_ctr_pts(img_width, points1)
        set2 = disp_pts_to_real_ctr_pts(img_width, points2)

        shift_sum = np.zeros(2, dtype=np.int32)
        for pt1, pt2 in zip(set1, set2):
            shift = np.array(pt1) - np.array(pt2)
            shift_sum += shift

        self.last_shift = list(shift_sum // len(points1))
        self.reshift()

    def auto_rotate_image(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1

        if curr_img.prev is None:
            print('Could not find the reference (preceding) image...')
            return

        points1 = self.point_sets[curr_idx-1]
        points2 = self.point_sets[curr_idx]

        if not validate_two_point_sets(points1, points2, min_sz=2):
            return

        np1 = len(points1)
        np2 = len(points2)

        if np1 % 2:
            np1 -= 1
            np2 -= 1
            points1 = points1[:-1]
            points2 = points2[:-1]

        rot_angles = []
        rot_angle_avg = 0.0
        n_pairs = np1 // 2

        for l in range(n_pairs):
            p11, p12 = points1[2*l:2*(l+1)]
            p21, p22 = points2[2*l:2*(l+1)]
            ang1 = tr.find_dir_angle(p11, p12)
            ang2 = tr.find_dir_angle(p21, p22)
            rot_angle = imsup.degrees(ang2 - ang1)
            rot_angles.append(rot_angle)
            rot_angle_avg += rot_angle

        rot_angle_avg /= n_pairs
        self.last_rot_angle = rot_angle_avg

        print('Partial rot. angles: ' + ', '.join('{0:.2f} deg'.format(ang) for ang in rot_angles))
        # print('Average rot. angle = {0:.2f} deg'.format(rot_angle_avg))
        self.rerotate()

    def reshift(self):
        print('Shifting by [dx, dy] = {0} px'.format(self.last_shift))
        curr_img = self.display.image
        shifted_img = imsup.shift_amph_image(curr_img, self.last_shift)
        shifted_img.name = curr_img.name + '_sh'
        self.insert_img_after_curr(shifted_img)

    def rerotate(self):
        print('Using rot. angle = {0:.2f} deg'.format(self.last_rot_angle))
        curr_img = self.display.image
        rotated_img = tr.rotate_image_ski(curr_img, self.last_rot_angle)
        rotated_img.name = curr_img.name + '_rot'
        self.insert_img_after_curr(rotated_img)

    def get_scale_ratio_from_images(self):
        if self.display.image.prev is None:
            print('Could not find the reference (preceding) image...')
            return
        curr_px_dim = self.display.image.px_dim
        prev_px_dim = self.display.image.prev.px_dim
        scale_ratio = curr_px_dim / prev_px_dim
        print('Scale ratio (between current and previous images) = {0:.2f}x'.format(scale_ratio))
        print('scale factor < 0 -- current image should be scaled')
        print('scale factor > 0 -- previous image should be scaled')
        self.scale_factor_input.setText('{0:.2f}'.format(scale_ratio))

    def scale_image(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1
        img_width = curr_img.width

        points2 = self.point_sets[curr_idx]
        np2 = len(points2)

        if curr_img.prev is None or np2 == 0:
            print('Manual scaling')
            self.last_scale_factor = float(self.scale_factor_input.text())
            self.rescale_image()
            return

        points1 = self.point_sets[curr_idx - 1]

        if not validate_two_point_sets(points1, points2, min_sz=2):
            return

        poly1 = disp_pts_to_real_ctr_pts(img_width, points1)
        poly2 = disp_pts_to_real_ctr_pts(img_width, points2)

        poly1_dists = []
        poly2_dists = []
        for i in range(len(poly1)):
            for j in range(i + 1, len(poly1)):
                poly1_dists.append(tr.calc_distance(poly1[i], poly1[j]))
                poly2_dists.append(tr.calc_distance(poly2[i], poly2[j]))

        scfs = [ dist1 / dist2 for dist1, dist2 in zip(poly1_dists, poly2_dists) ]
        scf_avg = np.average(scfs)
        self.last_scale_factor = scf_avg

        print('Automatic scaling:')
        print('Partial scale factors: ' + ', '.join('{0:.2f}x'.format(scf) for scf in scfs))
        # print('Average scale factor = {0:.2f}x'.format(scf_avg))
        self.rescale_image()

    def rescale_image(self):
        print('Using scale factor = {0:.2f}x'.format(self.last_scale_factor))
        curr_img = self.display.image
        mag_img = tr.rescale_image_ski(curr_img, self.last_scale_factor)
        pad_sz = (mag_img.width - curr_img.width) // 2

        if pad_sz > 0:
            crop_coords = 2 * [pad_sz] + 2 * [pad_sz + curr_img.width]
            resc_img = imsup.crop_amph_roi(mag_img, crop_coords)
        else:
            resc_img = imsup.pad_image(mag_img, curr_img.width, curr_img.height, pval=0.0)

        resc_img.name = curr_img.name + '_resc'
        self.insert_img_after_curr(resc_img)
        print('Image rescaled!')

    def warp_image(self, more_accurate=False):
        curr_img = self.display.image
        curr_idx = self.display.image.num_in_ser - 1

        if curr_img.prev is None:
            print('Could not find the reference (preceding) image...')
            return

        points1 = self.point_sets[curr_idx-1]
        points2 = self.point_sets[curr_idx]

        if not validate_two_point_sets(points1, points2, min_sz=4):
            return

        real_points1 = disp_pts_to_real_ctr_pts(curr_img.width, points1)
        real_points2 = disp_pts_to_real_ctr_pts(curr_img.width, points2)
        user_points1 = real_ctr_pts_to_tl_pts(curr_img.width, real_points1)
        user_points2 = real_ctr_pts_to_tl_pts(curr_img.width, real_points2)

        if more_accurate:
            n_div = const.n_div_for_warp
            frag_dim_size = curr_img.width // n_div

            # points #1
            grid_points1 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points1 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points1 ]

            for pt1 in user_points1:
                closest_node = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt1 ]
                grid_points1 = [ pt1 if grid_node == closest_node else grid_node for grid_node in grid_points1 ]

            # points #2
            grid_points2 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points2 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points2 ]
            for pt2 in user_points2:
                closestNode = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt2 ]
                grid_points2 = [ pt2 if gridNode == closestNode else gridNode for gridNode in grid_points2 ]

            self.warp_points = [ grid_points1, grid_points2 ]

        else:
            self.warp_points = [ user_points1, user_points2 ]

        self.rewarp()

    def rewarp(self):
        curr_img = self.display.image

        if len(self.warp_points) == 0:
            print('No image has been warped yet...')
            return

        src = np.array(self.warp_points[0])
        dst = np.array(self.warp_points[1])

        warped_img = tr.warp_image_ski(curr_img, src, dst)
        warped_img.name = curr_img.name + '_warp'
        self.insert_img_after_curr(warped_img)
        print('Image warped!')

    def holo_fft(self):
        h_img = self.display.image
        h_fft = holo.holo_fft(h_img)
        h_fft.name = 'fft_of_{0}'.format(h_img.name)
        self.hide_bad_px_checkbox.setChecked(False)
        self.insert_img_after_curr(h_fft)
        self.log_scale_checkbox.setChecked(True)

    def holo_get_sideband(self):
        # general convention is (x, y), i.e. (col, row)
        h_fft = self.display.image
        pt_set = self.point_sets[h_fft.num_in_ser - 1]

        if len(pt_set) < 2:
            print('Mark the sideband first...')
            return

        aper_diam = abs(int(self.aperture_input.text()))
        smooth_width = abs(int(self.smooth_width_input.text()))

        if not validate_aperture(aper_diam, smooth_width, h_fft.width):
            return

        print('--------------------------')
        print('Hologram reconstruction (no reference hologram)')
        print('Input:\n"{0}" -- FFT of the object hologram (with selected sideband)'.format(h_fft.name))

        pt1, pt2 = pt_set[:2]
        pt1, pt2 = tr.convert_points_to_tl_br(pt1, pt2)
        rpt1, rpt2 = disp_pts_to_real_tl_pts(h_fft.width, [pt1, pt2])

        sband = np.copy(h_fft.amph.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])
        apply_subpx_shift = self.subpixel_shift_checkbox.isChecked()
        sband_xy = holo.find_sideband_center(sband, orig=rpt1, subpx=apply_subpx_shift)

        # --- temp. code ---
        px_dim = h_fft.px_dim
        img_dim = h_fft.width
        mid_x = img_dim // 2
        dx_dim = 1 / (px_dim * img_dim)
        sbx, sby = sband_xy[0] - mid_x, sband_xy[1] - mid_x
        sb_xy_comp = np.complex(sbx * dx_dim, sby * dx_dim)
        R = np.abs(sb_xy_comp)
        ang = imsup.degrees(np.angle(sb_xy_comp))
        print('R = {0:.3f} um-1\nAng = {1:.2f} deg'.format(R * 1e-6, ang))
        # ------------------

        mid = h_fft.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        sband_ctr_ap = holo.holo_get_sideband(h_fft, shift, ap_dia=aper_diam, smooth_w=smooth_width)
        sband_ctr_ap.name = 'sband_{0}'.format(h_fft.name)

        self.log_scale_checkbox.setChecked(True)
        self.insert_img_after_curr(sband_ctr_ap)

        print('Output:\n"{0}" -- cropped and centered sideband of the object hologram'.format(sband_ctr_ap.name))
        print('--------------------------')

    def holo_with_ref_get_sidebands(self):
        ref_h_fft = self.display.image
        obj_h_img = self.display.image.next

        if obj_h_img is None:
            print('Could not find the object hologram (should be next in the queue)...')
            return

        pt_set = self.point_sets[ref_h_fft.num_in_ser - 1]

        if len(pt_set) < 2:
            print('Mark the sideband first...')
            return

        aper_diam = abs(int(self.aperture_input.text()))
        smooth_width = abs(int(self.smooth_width_input.text()))

        if not validate_aperture(aper_diam, smooth_width, ref_h_fft.width):
            return

        print('--------------------------')
        print('Hologram reconstruction (with reference hologram)'
              '\nInput:'
              '\n"{0}" -- FFT of the reference hologram (with selected sideband)'
              '\n"{1}" -- object hologram'.format(ref_h_fft.name, obj_h_img.name))

        pt1, pt2 = pt_set[:2]
        pt1, pt2 = tr.convert_points_to_tl_br(pt1, pt2)
        rpt1, rpt2 = disp_pts_to_real_tl_pts(ref_h_fft.width, [pt1, pt2])

        sband = np.copy(ref_h_fft.amph.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])
        apply_subpx_shift = self.subpixel_shift_checkbox.isChecked()
        sband_xy = holo.find_sideband_center(sband, orig=rpt1, subpx=apply_subpx_shift)

        mid = ref_h_fft.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        ref_sband_ctr_ap = holo.holo_get_sideband(ref_h_fft, shift, ap_dia=aper_diam, smooth_w=smooth_width)

        obj_h_fft = holo.holo_fft(obj_h_img)
        obj_sband_ctr_ap = holo.holo_get_sideband(obj_h_fft, shift, ap_dia=aper_diam, smooth_w=smooth_width)

        ref_sband_ctr_ap.name = 'ref_sband'
        obj_sband_ctr_ap.name = 'obj_sband'

        self.log_scale_checkbox.setChecked(True)
        self.insert_img_after_curr(ref_sband_ctr_ap)
        self.insert_img_after_curr(obj_sband_ctr_ap)

        print('Output:'
              '\n"{0}" -- cropped and centered sideband of the reference hologram'
              '\n"{1}" -- cropped and centered sideband of the object hologram'.format(ref_sband_ctr_ap.name, obj_sband_ctr_ap.name))
        print('--------------------------')

    def holo_ifft(self):
        h_fft = self.display.image
        h_ifft = holo.holo_ifft(h_fft)
        h_ifft.name = 'ifft_of_{0}'.format(h_fft.name)
        self.log_scale_checkbox.setChecked(False)
        self.insert_img_after_curr(h_ifft)

    def rec_holo_with_ref_auto(self):
        ref_h_fft = self.display.image
        obj_h_img = self.display.image.next

        if obj_h_img is None:
            print('Could not find the object hologram (should be next in the queue)...')
            return

        pt_set = self.point_sets[ref_h_fft.num_in_ser - 1]

        if len(pt_set) < 2:
            print('Mark the sideband first...')
            return

        aper_diam = abs(int(self.aperture_input.text()))
        smooth_width = abs(int(self.smooth_width_input.text()))

        if not validate_aperture(aper_diam, smooth_width, ref_h_fft.width):
            return

        self.parent().show_status_bar_message('Working...', change_bkg=True)

        print('--------------------------')
        print('(Fast) Hologram reconstruction (with reference hologram)'
              '\nInput:'
              '\n"{0}" -- FFT of the reference hologram (with selected sideband)'
              '\n"{1}" -- object hologram'.format(ref_h_fft.name, obj_h_img.name))

        pt1, pt2 = pt_set[:2]
        pt1, pt2 = tr.convert_points_to_tl_br(pt1, pt2)
        rpt1, rpt2 = disp_pts_to_real_tl_pts(ref_h_fft.width, [pt1, pt2])

        sband = np.copy(ref_h_fft.amph.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])
        apply_subpx_shift = self.subpixel_shift_checkbox.isChecked()
        sband_xy = holo.find_sideband_center(sband, orig=rpt1, subpx=apply_subpx_shift)

        mid = ref_h_fft.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        ref_sband_ctr_ap = holo.holo_get_sideband(ref_h_fft, shift, ap_dia=aper_diam, smooth_w=smooth_width)

        obj_h_fft = holo.holo_fft(obj_h_img)
        obj_sband_ctr_ap = holo.holo_get_sideband(obj_h_fft, shift, ap_dia=aper_diam, smooth_w=smooth_width)

        rec_ref = holo.holo_ifft(ref_sband_ctr_ap)
        rec_obj = holo.holo_ifft(obj_sband_ctr_ap)

        # unwrapping
        new_ref_phs = tr.unwrap_phase(rec_ref.amph.ph)
        new_obj_phs = tr.unwrap_phase(rec_obj.amph.ph)
        rec_ref.amph.ph = np.copy(new_ref_phs)
        rec_obj.amph.ph = np.copy(new_obj_phs)

        rec_obj_corr = holo.calc_phase_diff(rec_ref, rec_obj)

        self.log_scale_checkbox.setChecked(False)

        if self.assign_rec_ph_to_obj_h_checkbox.isChecked():
            # obj_h_img.amph.ph[:] = rec_obj_corr.amph.ph
            obj_h_img.amph.ph = np.copy(rec_obj_corr.amph.ph)
            obj_h_img = rescale_image_buffer_to_window(obj_h_img, const.disp_dim)
            obj_h_img.name += '_rec'
            self.go_to_next_image()
            print('Output:\n"{0}" -- object hologram (amplitude) with reconstructed phase'.format(obj_h_img.name))
        else:
            rec_obj_corr = rescale_image_buffer_to_window(rec_obj_corr, const.disp_dim)
            rec_obj_corr.name = 'ph_from_{0}'.format(obj_h_img.name)
            self.insert_img_after_curr(rec_obj_corr)
            print('Output:\n"{0}" -- reconstructed amplitude/phase of the object hologram'.format(rec_obj_corr.name))

        print('--------------------------')
        self.phs_radio_button.setChecked(True)
        self.parent().show_status_bar_message('', change_bkg=True)

    def calc_phs_sum(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        if rec_holo1 is None:
            print('Two images are necessary (sum = curr. phase + prev. phase)')
            return

        phs_sum = holo.calc_phase_sum(rec_holo1, rec_holo2)
        phs_sum.name = 'sum_{0}+{1}'.format(rec_holo2.name, rec_holo1.name)
        phs_sum = rescale_image_buffer_to_window(phs_sum, const.disp_dim)
        self.insert_img_after_curr(phs_sum)

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        if rec_holo1 is None:
            print('Two images are necessary (diff = curr. phase - prev. phase)')
            return

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)
        phs_diff.name = 'diff_{0}-{1}'.format(rec_holo2.name, rec_holo1.name)
        phs_diff = rescale_image_buffer_to_window(phs_diff, const.disp_dim)
        self.insert_img_after_curr(phs_diff)

    def amplify_phase(self):
        curr_img = self.display.image
        amp_factor = float(self.amp_factor_input.text())

        phs_amplified = imsup.copy_amph_image(curr_img)
        phs_amplified.amph.ph *= amp_factor
        phs_amplified.update_cos_phase()

        n_dec = '0' if amp_factor.is_integer() else '1'
        phs_amplified.name = '{0}_x{1:.{2}f}'.format(curr_img.name, amp_factor, n_dec)

        phs_amplified = rescale_image_buffer_to_window(phs_amplified, const.disp_dim)
        self.insert_img_after_curr(phs_amplified)
        self.cos_phs_radio_button.setChecked(True)

    def add_radians(self):
        curr_img = self.display.image
        radians = float(self.radians_to_add_input.text())

        new_phs_img = imsup.copy_amph_image(curr_img)
        new_phs_img.amph.ph += radians
        new_phs_img.update_cos_phase()

        name_app = '_-' if radians < 0.0 else '_+'
        name_app += '{0:.2f}rad'.format(abs(radians))
        new_phs_img.name = '{0}{1}'.format(curr_img.name, name_app)

        new_phs_img = rescale_image_buffer_to_window(new_phs_img, const.disp_dim)
        self.insert_img_after_curr(new_phs_img)
        self.cos_phs_radio_button.setChecked(True)
        print('Added {0:.2f} rad to "{1}"'.format(radians, curr_img.name))

    def remove_phase_tilt(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1
        h, w = curr_img.amph.am.shape
        phs = np.copy(curr_img.amph.ph)

        # default points
        xy1 = [0, h//2]
        xy2 = [w-1, h//2]
        xy3 = [w//2, 0]
        xy4 = [w//2, h-1]

        n_usr_pts = len(self.point_sets[curr_idx])

        if n_usr_pts == 2:
            print('Removing local phase tilt...')
            dpt1, dpt2 = self.point_sets[curr_idx][:2]
            rpt1 = disp_pt_to_real_tl_pt(w, dpt1)
            rpt2 = disp_pt_to_real_tl_pt(w, dpt2)
            mid_x = (rpt1[0]+rpt2[0]) // 2
            mid_y = (rpt1[1]+rpt2[1]) // 2
            xy1 = [rpt1[0], mid_y]
            xy2 = [rpt2[0], mid_y]
            xy3 = [mid_x, rpt1[1]]
            xy4 = [mid_x, rpt2[1]]
        elif n_usr_pts == 4:
            print('Removing global phase tilt...')
            dpts = self.point_sets[curr_idx][:4]
            # dpts_f = [c for dpt in dpts for c in dpt]     # unpacking list of lists
            rpts = [ disp_pt_to_real_tl_pt(w, dpt) for dpt in dpts ]
            xy1[0] = rpts[0][0]
            xy2[0] = rpts[1][0]
            xy3[1] = rpts[2][1]
            xy4[1] = rpts[3][1]
        else:
            print('Using default configuration... [To change it mark 2 points (local phase tilt) or 4 points (global phase tilt) and repeat procedure]')

        nn_def = 10

        # -- linear least squares ---
        n_pts_for_ls = 10
        n_neigh_areas = 2 * (n_pts_for_ls - 1)
        x_dist, y_dist = xy2[0] - xy1[0], xy4[1] - xy3[1]
        n_neigh_x = nn_def if x_dist >= n_neigh_areas * nn_def else int(x_dist // n_neigh_areas)
        n_neigh_y = nn_def if y_dist >= n_neigh_areas * nn_def else int(y_dist // n_neigh_areas)

        x_arr_for_ls = np.round(np.linspace(xy1[0], xy2[0], n_pts_for_ls)).astype(np.int32)
        y_arr_for_ls = np.round(np.linspace(xy3[1], xy4[1], n_pts_for_ls)).astype(np.int32)
        phx_arr_for_ls = np.array([ tr.calc_avg_neigh(phs, x=x, y=xy1[1], nn=n_neigh_x) for x in x_arr_for_ls] )
        phy_arr_for_ls = np.array([ tr.calc_avg_neigh(phs, x=xy3[0], y=y, nn=n_neigh_y) for y in y_arr_for_ls] )
        ax, bx = tr.lin_least_squares_alt(x_arr_for_ls, phx_arr_for_ls)
        ay, by = tr.lin_least_squares_alt(y_arr_for_ls, phy_arr_for_ls)

        X = np.arange(0, w, dtype=np.float32)
        Y = np.arange(0, h, dtype=np.float32)
        X, Y = np.meshgrid(X, Y)
        phs_grad_x = ax * X
        phs_grad_y = ay * Y
        # ---

        # --- without least squares (just boundary points) ---
        # px1 = [xy1[0], tr.calc_avg_neigh(phs, x=xy1[0], y=xy1[1], nn=nn_def)]
        # px2 = [xy2[0], tr.calc_avg_neigh(phs, x=xy2[0], y=xy2[1], nn=nn_def)]
        # py1 = [xy3[1], tr.calc_avg_neigh(phs, x=xy3[0], y=xy3[1], nn=nn_def)]
        # py2 = [xy4[1], tr.calc_avg_neigh(phs, x=xy4[0], y=xy4[1], nn=nn_def)]
        #
        # x_line = tr.Line(0, 0)
        # y_line = tr.Line(0, 0)
        # x_line.get_from_2_points(px1, px2)
        # y_line.get_from_2_points(py1, py2)
        #
        # X = np.arange(0, w, dtype=np.float32)
        # Y = np.arange(0, h, dtype=np.float32)
        # X, Y = np.meshgrid(X, Y)
        # phs_grad_x = x_line.a * X # + x_line.b
        # phs_grad_y = y_line.a * Y # + y_line.b
        # ---

        phs_grad_xy = phs_grad_x + phs_grad_y

        phs_grad_img = imsup.ImageExp(curr_img.height, curr_img.width)
        phs_grad_img.load_phs_data(phs_grad_xy)
        phs_grad_img.name = '{0}_tilt'.format(curr_img.name)

        new_phs_img = imsup.copy_amph_image(curr_img)
        new_phs_img.amph.ph -= phs_grad_xy
        new_phs_img.name = '{0}_--tilt'.format(curr_img.name)

        self.insert_img_after_curr(phs_grad_img)
        self.insert_img_after_curr(new_phs_img)

    def get_sideband_from_xy(self):
        h_fft = self.display.image

        aper_diam = abs(int(self.aperture_input.text()))
        smooth_width = abs(int(self.smooth_width_input.text()))

        if not validate_aperture(aper_diam, smooth_width, h_fft.width):
            return

        sbx = float(self.sideband_x_input.text())
        sby = float(self.sideband_y_input.text())

        mid = h_fft.width // 2
        shift = [mid - sbx, mid - sby]

        sband_ctr_ap = holo.holo_get_sideband(h_fft, shift, ap_dia=aper_diam, smooth_w=smooth_width)
        sband_ctr_ap.name = 'sband_{0}'.format(h_fft.name)

        self.log_scale_checkbox.setChecked(True)
        self.insert_img_after_curr(sband_ctr_ap)

    def plot_profile(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1
        px_sz = curr_img.px_dim

        dpts = self.point_sets[curr_idx][:2]
        rpts = np.array(disp_pts_to_real_ctr_pts(curr_img.width, dpts))
        rpt1, rpt2 = rpts

        # find rotation center of the section
        rot_center = np.average(rpts, 0).astype(np.int32)

        # find direction angle of the section
        dir_angle = tr.find_dir_angle(rpt1, rpt2)

        # shift rotation center to the image center
        shift_to_rc = list(-rot_center)
        img_shifted = imsup.shift_amph_image(curr_img, shift_to_rc)

        # rotate image by the direction angle
        rot_angle = imsup.degrees(dir_angle)
        img_rotated = tr.rotate_image_ski(img_shifted, rot_angle, resize=True)

        # crop fragment (width = distance between two points, height = profile width)
        frag_w = int(np.linalg.norm(rpt1 - rpt2))
        frag_h = int(self.prof_width_input.text())

        if frag_h <= 0 or frag_h > img_rotated.height:
            print('Incorrect profile width')
            return

        frag_coords = imsup.det_crop_coords_for_new_dims(img_rotated.width, img_rotated.height, frag_w, frag_h)
        img_cropped = imsup.crop_amph_roi(img_rotated, frag_coords)

        # calculate projection of intensity
        if self.amp_radio_button.isChecked():
            int_matrix = np.copy(img_cropped.amph.am)
        elif self.phs_radio_button.isChecked():
            int_matrix = np.copy(img_cropped.amph.ph)
        else:
            int_matrix = np.copy(img_cropped.cos_phase)

        int_profile = np.sum(int_matrix, 0) / frag_h    # 0 - horizontal projection, 1 - vertical projection
        dists = np.arange(0, int_profile.shape[0], 1) * px_sz
        dists *= 1e9

        self.plot_widget.plot(dists, int_profile, 'Distance [nm]', 'Intensity [a.u.]')

    def calc_phase_gradient(self):
        curr_img = self.display.image
        dx_img = imsup.ImageExp(curr_img.height, curr_img.width)
        dy_img = imsup.ImageExp(curr_img.height, curr_img.width)
        grad_img = imsup.ImageExp(curr_img.height, curr_img.width)
        print('Calculating gradient for sample distance = {0:.2f} nm'.format(curr_img.px_dim * 1e9))
        dx, dy = np.gradient(curr_img.amph.ph, curr_img.px_dim)
        dr = np.sqrt(dx * dx + dy * dy)
        # dphi = np.arctan2(dy, dx)
        dx_img.amph.ph = np.copy(dx)
        dy_img.amph.ph = np.copy(dy)
        grad_img.amph.ph = np.copy(dr)
        dx_img.name = 'gradX_of_{0}'.format(curr_img.name)
        dy_img.name = 'gradY_of_{0}'.format(curr_img.name)
        grad_img.name = 'gradM_of_{0}'.format(curr_img.name)
        self.insert_img_after_curr(dx_img)
        self.insert_img_after_curr(dy_img)
        self.insert_img_after_curr(grad_img)

    # calculate B from section on image
    def calc_B_from_section(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1

        dpt1, dpt2 = self.point_sets[curr_idx][:2]
        pt1 = np.array(disp_pt_to_real_tl_pt(curr_img.width, dpt1))
        pt2 = np.array(disp_pt_to_real_tl_pt(curr_img.width, dpt2))

        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        mc.calc_B_from_section(curr_img, pt1, pt2, sample_thickness)

    # calculate B from profile in PlotWidget
    def calc_B_from_profile(self):
        if len(self.plot_widget.marked_points) < 2:
            print('You have to mark two points on the profile...')
            return
        pt1, pt2 = self.plot_widget.marked_points_data
        d_dist_real = np.abs(pt1[0] - pt2[0]) * 1e-9
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        mc.calc_B(pt1[1], pt2[1], d_dist_real, sample_thickness, print_msg=True)

    def calc_Bxy_maps(self):
        curr_img = self.display.image
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        Bx_img, By_img = mc.calc_Bxy_maps(curr_img, sample_thickness)
        self.insert_img_after_curr(Bx_img)
        self.insert_img_after_curr(By_img)

    def calc_B_polar_from_section(self, gen_multiple_plots=False):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1

        if len(self.point_sets[curr_idx]) < 2:
            print('You have to mark two points on the image...')
            return

        dpt1, dpt2 = self.point_sets[curr_idx][:2]
        pt1 = np.array(disp_pt_to_real_tl_pt(curr_img.width, dpt1))
        pt2 = np.array(disp_pt_to_real_tl_pt(curr_img.width, dpt2))
        d_dist = np.linalg.norm(pt1 - pt2)

        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        orig_is_pt1 = self.orig_in_pt1_radio_button.isChecked()
        n_r_iters = int(self.num_of_r_iters_input.text())
        n_rows = int(self.B_pol_n_rows_input.text())
        n_cols = int(self.B_pol_n_cols_input.text())

        if orig_is_pt1:
            orig_xy = pt1
            r1 = d_dist
        else:
            orig_xy = np.round(np.mean([pt1, pt2], axis=0)).astype(np.int32)
            r1 = d_dist / 2

        if gen_multiple_plots is False:
            dir_ang = -np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            mc.calc_B_polar_from_orig_r(curr_img, orig_xy, r1, sample_thickness, orig_is_pt1, dir_ang, n_r_iters)
        else:
            orig_pts = mc.calc_B_polar_sectors(curr_img, orig_xy, r1, n_rows, n_cols, sample_thickness, orig_is_pt1,
                                               n_r_iters)
            d_orig_pts = real_tl_pts_to_disp_pts(curr_img.width, orig_pts)
            self.point_sets[curr_idx].extend(d_orig_pts)
            self.display.repaint()
            if self.display.show_labs:
                self.display.show_labels()
            print('Orig. points were added to the display')

    def calc_B_polar_from_area(self):
        curr_img = self.display.image
        curr_idx = curr_img.num_in_ser - 1

        if len(self.point_sets[curr_idx]) < 2:
            print('You have to mark two points on the image...')
            return

        pt1, pt2 = self.point_sets[curr_idx][:2]
        pt1, pt2 = tr.convert_points_to_tl_br(pt1, pt2)
        disp_crop_coords = pt1 + pt2
        real_tl_coords = disp_pt_to_real_tl_pt(curr_img.width, disp_crop_coords)
        real_sq_coords = imsup.make_square_coords(real_tl_coords)
        frag = crop_fragment(curr_img, real_sq_coords)
        frag.name = curr_img.name

        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        mc.calc_B_polar_from_area(frag, sample_thickness)

    def gen_phase_stats(self):
        curr_img = self.display.image
        curr_phs = curr_img.amph.ph
        print('STATISTICS for phase of "{0}":'.format(curr_img.name))
        print('Min. = {0:.2f}\nMax. = {1:.2f}\nAvg. = {2:.2f}'.format(np.min(curr_phs), np.max(curr_phs), np.mean(curr_phs)))
        print('Med. = {0:.2f}\nStd. dev. = {1:.2f}\nVar. = {2:.2f}'.format(np.median(curr_phs), np.std(curr_phs), np.var(curr_phs)))

        # if curr_img.prev is not None:
        #     max_shift = const.corr_arr_max_shift
        #     print('Calculating ({0}x{0}) correlation array between *curr* and *prev* phases...'.format(2 * max_shift))
        #     prev_img = curr_img.prev
        #     prev_phs = prev_img.amph.ph
        #     corr_arr = tr.calc_corr_array(prev_phs, curr_phs, max_shift)
        #     ch, cw = corr_arr.shape
        #     corr_img = imsup.ImageExp(ch, cw)
        #     corr_img.load_amp_data(corr_arr)
        #     corr_img.load_phs_data(corr_arr)
        #     corr_img = rescale_image_buffer_to_window(corr_img, const.disp_dim)
        #     corr_img.name = 'corr_arr_{0}_vs_{1}'.format(curr_img.name, prev_img.name)
        #     self.insert_img_after_curr(corr_img)
        #
        #     # # single-value correlation coefficient
        #     # p1 = prev_phs - np.mean(prev_phs)
        #     # p2 = curr_phs - np.mean(curr_phs)
        #     # corr_coef = np.sum(p1 * p2) / np.sqrt(np.sum(p1 * p1) * np.sum(p2 * p2))
        #     # print('Corr. coef. between *curr* and *prev* phases = {0:.4f}'.format(corr_coef))
        #     #
        #     # # Pearson correlation matrix
        #     # corr_coef_arr = np.corrcoef(prev_phs, curr_phs)
        #     # corr_coef_img = imsup.ImageExp(curr_img.height, curr_img.width, px_dim_sz=curr_img.px_dim)
        #     # corr_coef_img.amph.ph = np.copy(corr_coef_arr)
        #     # corr_coef_img = rescale_image_buffer_to_window(corr_coef_img, const.disp_dim)
        #     # corr_coef_img.name = 'corr_coef_{0}_vs_{1}'.format(curr_img.name, prev_img.name)
        #     # self.insert_img_after_curr(corr_coef_img)

    def calc_mean_inner_potential(self):
        curr_img = self.display.image
        curr_phs = curr_img.amph.ph

        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        Ua = float(self.acc_voltage_input.text()) * 1e3

        h = const.planck_const
        c = const.light_speed
        e0 = const.el_charge
        m0 = const.el_rest_mass
        eps = e0 / (2 * m0 * c ** 2)
        ew_lambda = h / np.sqrt(2 * m0 * e0 * Ua * (1 + eps * Ua))
        U0 = m0 * (c ** 2) / e0
        C_E = (2 * np.pi / ew_lambda) * (Ua + U0) / (Ua * (Ua + 2 * U0))
        mip = curr_phs / (C_E * sample_thickness)

        mean_inner_pot_img = imsup.copy_amph_image(curr_img)
        mean_inner_pot_img.amph.am *= 0
        mean_inner_pot_img.amph.ph = np.copy(mip)
        mean_inner_pot_img.name = 'MIP_from_{0}'.format(curr_img.name)
        self.insert_img_after_curr(mean_inner_pot_img)

    def filter_contours(self):
        curr_img = self.display.image
        curr_img.update_cos_phase()
        conts = np.copy(curr_img.cos_phase)
        conts_scaled = imsup.scale_image(conts, 0.0, 1.0)
        threshold = float(self.threshold_input.text())
        conts_scaled[conts_scaled < threshold] = 0.0
        img_filtered = imsup.copy_amph_image(curr_img)
        img_filtered.amph.ph = np.copy(conts_scaled)
        self.insert_img_after_curr(img_filtered)

# --------------------------------------------------------

def open_dm3_file(file_path, img_type='amp'):
    img_data, px_dims = dm3.ReadDm3File(file_path)
    imsup.Image.px_dim_default = px_dims[0]
    new_img = imsup.ImageExp(img_data.shape[0], img_data.shape[1], imsup.Image.cmp['CAP'], px_dim_sz=px_dims[0])

    if img_type == 'amp':
        # amp_data = np.sqrt(np.abs(img_data))
        amp_data = np.copy(img_data)
        new_img.load_amp_data(amp_data.astype(np.float32))
    else:
        new_img.load_phs_data(img_data.astype(np.float32))

    return new_img

# --------------------------------------------------------

def load_image_series_from_first_file(img_fpath):
    img_list = imsup.ImageList()
    img_num_match = re.search('([0-9]+).dm3', img_fpath)

    if img_num_match is None:
        print('Invalid file name (number is missing).\n'
              'Rename files to match the following convention: "ser1.dm3", "ser2.dm3", "ser3.dm3", ...\n'
              'The part before the number has to be the same for all files in that series.')
        return None             # ignore the file with invalid name and return to the main program
    #     img_num_text = '1'    # read the file with invalid name anyway
    # else:
    #     img_num_text = img_num_match.group(1)

    img_num_text = img_num_match.group(1)
    img_num = int(img_num_text)

    img_idx = 0
    info = None
    name_col, type_col = 1, 2
    valid_types = ('amp', 'phs')

    if img_idx == 0:
        first_img_name_match = re.search('(.+)/(.+).dm3$', img_fpath)
        dir_path = first_img_name_match.group(1)
        info_fpath = '{0}/info.txt'.format(dir_path)
        if path.isfile(info_fpath) and path.getsize(info_fpath) > 0:
            print('info file detected')
            import pandas as pd
            info = pd.read_csv(info_fpath, sep='\t', header=None)
            first_row_idx = img_num - 1 if img_num > 0 else 0
            info = info.values[first_row_idx:, :]
        else:
            print('info file not detected or empty')

    while path.isfile(img_fpath):
        print('Reading file "' + img_fpath + '"')
        img_fname_match = re.search('(.+)/(.+).dm3$', img_fpath)
        img_fname = img_fname_match.group(2)
        img_name = img_fname
        img_type = valid_types[0]

        if info is not None and img_idx < info.shape[0]:
            if info.shape[1] > name_col:
                img_name = info[img_idx, name_col]
            if info.shape[1] > type_col and info[img_idx, type_col] in valid_types:
                img_type = info[img_idx, type_col]

        img = open_dm3_file(img_fpath, img_type)
        img.num_in_ser = img_num
        img.name = img_name
        img = rescale_image_buffer_to_window(img, const.disp_dim)
        img_list.append(img)

        img_idx += 1
        img_num += 1

        img_num_text_new = img_num_text.replace(str(img_num-1), str(img_num))
        if img_num == 10 and img_num_text_new[0] == '0':
            img_num_text_new = img_num_text_new[1:]

        img_fname_new = rreplace(img_fname, img_num_text, img_num_text_new, 1)
        img_fpath = rreplace(img_fpath, img_fname, img_fname_new, 1)
        img_num_text = img_num_text_new

    img_list.update_links()
    # img_list.update_and_restrain_links()
    return img_list[0]

# --------------------------------------------------------

def rescale_image_buffer_to_window(img, win_dim):
    scale_factor = win_dim / img.width
    img_to_disp = tr.rescale_image_ski(img, scale_factor)
    img.buffer = imsup.ComplexAmPhMatrix(img_to_disp.height, img_to_disp.width)
    img.buffer.am = np.copy(img_to_disp.amph.am)
    img.buffer.ph = np.copy(img_to_disp.amph.ph)
    return img

# --------------------------------------------------------

def cross_corr_images(img_list):
    img_align_list = imsup.ImageList()
    img_list[0].shift = [0, 0]
    for img in img_list[1:]:
        mcf = imsup.calc_cross_corr_fun(img.prev, img)
        new_shift = imsup.get_shift(mcf)
        img.shift = [ sp + sn for sp, sn in zip(img.prev.shift, new_shift) ]
        # img.shift = list(np.array(img.shift) + np.array(new_shift))
        print('"{0}" was shifted by {1} px'.format(img.name, img.shift))
        img_shifted = imsup.shift_amph_image(img, img.shift)
        img_align_list.append(img_shifted)
    return img_align_list

# --------------------------------------------------------

def crop_fragment(img, coords):
    crop_img = imsup.crop_amph_roi(img, coords)
    crop_img = rescale_image_buffer_to_window(crop_img, const.disp_dim)
    return crop_img

# --------------------------------------------------------

def validate_two_point_sets(pts1, pts2, min_sz=0):
    np1, np2 = len(pts1), len(pts2)
    if np1 < min_sz or np1 != np2:
        print('Mark the same (>={0}) number of reference points on both images...'.format(min_sz))
        return 0
    return 1

# --------------------------------------------------------

def validate_aperture(ap_dia, smooth_w, img_dim):
    if ap_dia + 2 * smooth_w > img_dim:
        print('Aperture exceeds the image size')
        return 0
    return 1

# --------------------------------------------------------

def real_ctr_pt_to_tl_pt(img_width, center_pt):
    top_left_pt = [ cc + img_width // 2 for cc in center_pt ]
    return top_left_pt

# --------------------------------------------------------

def real_ctr_pts_to_tl_pts(img_width, center_pts):
    top_left_pts = [ real_ctr_pt_to_tl_pt(img_width, cpt) for cpt in center_pts ]
    return top_left_pts

# --------------------------------------------------------

def disp_pt_to_real_tl_pt(img_width, disp_pt):
    disp_width = const.disp_dim
    factor = img_width / disp_width
    real_pt = [ int(round(dc * factor)) for dc in disp_pt ]
    return real_pt

# --------------------------------------------------------

def disp_pts_to_real_tl_pts(img_width, disp_pts):
    real_pts = [ disp_pt_to_real_tl_pt(img_width, dpt) for dpt in disp_pts ]
    return real_pts

# --------------------------------------------------------

def disp_pt_to_real_ctr_pt(img_width, disp_pt):
    disp_width = const.disp_dim
    factor = img_width / disp_width
    real_pt = [ int(round((dc - disp_width // 2) * factor)) for dc in disp_pt ]
    return real_pt

# --------------------------------------------------------

def disp_pts_to_real_ctr_pts(img_width, disp_pts):
    real_pts = [ disp_pt_to_real_ctr_pt(img_width, dpt) for dpt in disp_pts ]
    return real_pts

# --------------------------------------------------------

def real_tl_pt_to_disp_pt(img_width, real_pt):
    disp_width = const.disp_dim
    factor = disp_width / img_width
    disp_pt = [ int(round(rc * factor)) for rc in real_pt ]
    return disp_pt

# --------------------------------------------------------

def real_tl_pts_to_disp_pts(img_width, real_pts):
    disp_pts = [ real_tl_pt_to_disp_pt(img_width, rpt) for rpt in real_pts ]
    return disp_pts

# --------------------------------------------------------

def real_ctr_pt_to_disp_pt(img_width, real_pt):
    disp_width = const.disp_dim
    factor = disp_width / img_width
    disp_pt = [ int(round(rc * factor)) + disp_width // 2 for rc in real_pt ]
    return disp_pt

# --------------------------------------------------------

def real_ctr_pts_to_disp_pts(img_width, real_pts):
    disp_pts = [ real_ctr_pt_to_disp_pt(img_width, rpt) for rpt in real_pts ]
    return disp_pts

# --------------------------------------------------------

def switch_xy(xy):
    return [xy[1], xy[0]]

# --------------------------------------------------------

def rreplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

# def plot_arrow_fun(x, y, dx, dy, sc=1):
#     plt.arrow(x, y, dx*sc, dy*sc, fc="k", ec="k", lw=1.0, head_width=10, head_length=14)

# --------------------------------------------------------

def plot_arrow_fun(ax, x, y, dx, dy, sc=1):
    # ax.arrow(x, y, dx*sc, dy*sc, fc="k", ec="k", lw=1.0, head_width=10, head_length=14)
    ax.arrow(x, y, dx*sc, dy*sc, fc="k", ec="k", lw=0.6, head_width=5, head_length=8)

# --------------------------------------------------------

def export_glob_sc_images(img_list, add_arrows=True, rot_by_90=False, arr_size=20, arr_dist=50, cbar_lab=''):
    global_limits = [1e5, 0]

    for img in img_list:
        limits = [np.min(img.amph.ph), np.max(img.amph.ph)]
        if limits[0] < global_limits[0]:
            global_limits[0] = limits[0]
        if limits[1] > global_limits[1]:
            global_limits[1] = limits[1]

    fig, ax = plt.subplots()
    for img, idx in zip(img_list, range(1, len(img_list)+1)):
        im = ax.imshow(img.amph.ph, vmin=global_limits[0], vmax=global_limits[1], cmap=plt.cm.get_cmap('jet'))

        if idx == len(img_list):
            cbar = fig.colorbar(im)
            cbar.set_label(cbar_lab)

        if add_arrows:
            width, height = img.amph.ph.shape
            xv, yv = np.meshgrid(np.arange(0, width, float(arr_dist)), np.arange(0, height, float(arr_dist)))
            xv += arr_dist / 2.0
            yv += arr_dist / 2.0

            phd = img.amph.ph[0:height:arr_dist, 0:width:arr_dist]
            yd, xd = np.gradient(phd)

            # arrows along magnetic contours
            if rot_by_90:
                xd_yd_comp = xd + 1j * yd
                xd_yd_comp_rot = xd_yd_comp * np.exp(1j * np.pi / 2.0)
                xd = xd_yd_comp_rot.real
                yd = xd_yd_comp_rot.imag

            vectorized_arrow_drawing = np.vectorize(plot_arrow_fun)
            vectorized_arrow_drawing(ax, xv, yv, xd, yd, arr_size)

        out_f = '{0}.png'.format(img.name)
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(out_f, dpi=300, bbox_inches='tight', pad_inches=0)
        ax.cla()

    fig.clf()
    plt.close(fig)

# --------------------------------------------------------

def run_holography_window():
    app = QtWidgets.QApplication(sys.argv)
    holo_window = HolographyWindow()
    sys.exit(app.exec_())