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
import Holo as holo

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# --------------------------------------------------------

def func_to_vectorize(x, y, dx, dy, sc=1):
    plt.arrow(x, y, dx*sc, dy*sc, fc="k", ec="k", lw=0.6, head_width=10, head_length=14)
    # plt.arrow(x, y, dx * sc, dy * sc, fc="k", ec="k", lw=0.6, head_width=5, head_length=8)

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
            px_arr = np.copy(self.image.amPh.am)
        else:
            px_arr = np.copy(self.image.amPh.ph)

        pixmap = imsup.ScaleImage(px_arr, 0.0, 255.0)
        q_image = QtGui.QImage(pixmap.astype(np.uint8), pixmap.shape[0], pixmap.shape[1], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(q_image)
        self.setPixmap(pixmap)
        self.repaint()

# --------------------------------------------------------

class LabelExt(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        self.image = image
        self.setImage()
        self.pointSets = [[]]
        self.show_lines = True
        self.show_labs = True
        self.rgb_cm = RgbColorTable_B2R()

    # prowizorka - stałe liczbowe do poprawy
    def paintEvent(self, event):
        super(LabelExt, self).paintEvent(event)
        linePen = QtGui.QPen(QtCore.Qt.yellow)
        linePen.setCapStyle(QtCore.Qt.RoundCap)
        linePen.setWidth(3)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        imgIdx = self.image.numInSeries - 1
        qp.setPen(linePen)
        qp.setBrush(QtCore.Qt.yellow)

        for pt in self.pointSets[imgIdx]:
            # rect = QtCore.QRect(pt[0]-3, pt[1]-3, 7, 7)
            # qp.drawArc(rect, 0, 16*360)
            qp.drawEllipse(pt[0]-3, pt[1]-3, 7, 7)

        linePen.setWidth(2)
        if self.show_lines:
            qp.setPen(linePen)
            for pt1, pt2 in zip(self.pointSets[imgIdx], self.pointSets[imgIdx][1:] + self.pointSets[imgIdx][:1]):
                line = QtCore.QLine(pt1[0], pt1[1], pt2[0], pt2[1])
                qp.drawLine(line)

        linePen.setStyle(QtCore.Qt.DashLine)
        linePen.setColor(QtCore.Qt.yellow)
        linePen.setCapStyle(QtCore.Qt.FlatCap)
        qp.setPen(linePen)
        qp.setBrush(QtCore.Qt.NoBrush)
        if len(self.pointSets[imgIdx]) == 2:
            pt1, pt2 = self.pointSets[imgIdx]
            pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
            w = np.abs(pt2[0] - pt1[0])
            h = np.abs(pt2[1] - pt1[1])
            rect = QtCore.QRect(pt1[0], pt1[1], w, h)
            qp.drawRect(rect)
            sq_coords = imsup.MakeSquareCoords(pt1 + pt2)
            sq_pt1 = sq_coords[:2]
            sq_pt2 = sq_coords[2:]
            w = np.abs(sq_pt2[0]-sq_pt1[0])
            h = np.abs(sq_pt2[1]-sq_pt1[1])
            square = QtCore.QRect(sq_pt1[0], sq_pt1[1], w, h)
            linePen.setColor(QtCore.Qt.red)
            qp.setPen(linePen)
            qp.drawRect(square)
        qp.end()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        curr_pos = [pos.x(), pos.y()]
        self.pointSets[self.image.numInSeries - 1].append(curr_pos)
        self.repaint()

        pt_idx = len(self.pointSets[self.image.numInSeries - 1])
        real_x, real_y = CalcRealTLCoords(self.image.width, curr_pos)
        print('Added point {0} at:\nx = {1}\ny = {2}'.format(pt_idx, pos.x(), pos.y()))
        print('Actual position:\nx = {0}\ny = {1}'.format(real_x, real_y))
        print('Amp = {0:.2f}\nPhs = {1:.2f}'.format(self.image.amPh.am[real_y, real_x], self.image.amPh.ph[real_y, real_x]))

        if self.show_labs:
            lab = QtWidgets.QLabel('{0}'.format(pt_idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pos.x()+4, pos.y()+4)
            lab.show()

    def setImage(self, dispAmp=True, dispPhs=False, logScale=False, color=False, update_bcg=False, bright=0, cont=255, gamma=1.0):
        if self.image.buffer.am.shape[0] == self.image.height:
            self.image = rescale_image_buffer_to_window(self.image, const.disp_dim)

        if dispAmp:
            px_arr = np.copy(self.image.buffer.am)
            if logScale:
                buf_am = np.copy(px_arr)
                buf_am[np.where(buf_am <= 0)] = 1e-5
                px_arr = np.log(buf_am)
        else:
            px_arr = np.copy(self.image.buffer.ph)
            if not dispPhs:
                self.image.update_cos_phase()
                px_arr = np.cos(px_arr)

        if not update_bcg:
            pixmap_to_disp = imsup.ScaleImage(px_arr, 0.0, 255.0)
        else:
            pixmap_to_disp = update_image_bright_cont_gamma(px_arr, brg=bright, cnt=cont, gam=gamma)

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
        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

    def show_labels(self):
        img_idx = self.image.numInSeries - 1
        n_pt = len(self.pointSets[img_idx])
        for pt, idx in zip(self.pointSets[img_idx], range(1, n_pt+1)):
            lab = QtWidgets.QLabel('{0}'.format(idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pt[0] + 4, pt[1] + 4)
            lab.show()

    def show_last_label(self):
        img_idx = self.image.numInSeries - 1
        pt_idx = len(self.pointSets[img_idx]) - 1
        last_pt = self.pointSets[img_idx][pt_idx]
        lab = QtWidgets.QLabel('{0}'.format(pt_idx+1), self)
        lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
        lab.move(last_pt[0] + 4, last_pt[1] + 4)
        lab.show()

    def update_labels(self):
        # if len(self.pointSets) < self.image.numInSeries:
        #     self.pointSets.append([])
        self.hide_labels()
        if self.show_labs:
            self.show_labels()

# --------------------------------------------------------

def update_image_bright_cont_gamma(img_src, brg=0, cnt=1, gam=1.0):
    Imin, Imax = det_Imin_Imax_from_contrast(cnt)

    # option 1 (c->b->g)
    # correct contrast
    img_scaled = imsup.ScaleImage(img_src, Imin, Imax)
    # correct brightness
    img_scaled += brg
    img_scaled[img_scaled < 0.0] = 0.0
    # correct gamma
    img_scaled **= gam
    img_scaled[img_scaled > 255.0] = 255.0

    # # option 2 (c->g->b)
    # # correct contrast
    # img_scaled = imsup.ScaleImage(img_src, Imin, Imax)
    # img_scaled[img_scaled < 0.0] = 0.0
    # # correct gamma
    # img_scaled **= gam
    # # correct brightness
    # img_scaled += brg
    # img_scaled[img_scaled < 0.0] = 0.0
    # img_scaled[img_scaled > 255.0] = 255.0
    return img_scaled

# --------------------------------------------------------

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.markedPoints = []
        self.markedPointsData = []
        self.canvas.mpl_connect('button_press_event', self.getXYDataOnClick)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, dataX, dataY, xlab='x', ylab='y'):
        self.figure.clear()
        self.markedPoints = []
        self.markedPointsData = []
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axis([ min(dataX)-0.5, max(dataX)+0.5, min(dataY)-0.5, max(dataY)+0.5 ])
        ax = self.figure.add_subplot(111)
        ax.plot(dataX, dataY, '.-')
        self.canvas.draw()

    def getXYDataOnClick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if len(self.markedPoints) == 2:
            for pt in self.markedPoints:
                pt.remove()
            self.markedPoints = []
            self.markedPointsData = []
        pt, = plt.plot(event.xdata, event.ydata, 'ro')
        print(event.xdata, event.ydata)
        self.markedPoints.append(pt)
        self.markedPointsData.append([event.xdata, event.ydata])

# --------------------------------------------------------

class LineEditWithLabel(QtWidgets.QWidget):
    def __init__(self, parent, lab_text='Label', default_input=''):
        super(LineEditWithLabel, self).__init__(parent)
        self.label = QtWidgets.QLabel(lab_text)
        self.input = QtWidgets.QLineEdit(default_input)
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

        first_img = imsup.GetFirstImage(any_img)
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        if len(img_list) > 0:
            for img in img_list:
                preview_img = create_preview_img(img, (64, 64))
                preview = SimpleImageLabel(preview_img)
                self.scroll_layout.addWidget(preview)

# --------------------------------------------------------

def create_preview_img(full_img, new_sz):
    sx, sy = new_sz
    preview = imsup.ImageExp(sx, sy, full_img.cmp)
    preview.amPh.am = np.copy(full_img.amPh.am[:sx, :sy])
    preview.amPh.ph = np.copy(full_img.amPh.ph[:sx, :sy])
    return preview

# --------------------------------------------------------

class HolographyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(HolographyWindow, self).__init__()
        self.holo_widget = HolographyWidget()

        # ------------------------------
        # Menu bar
        # ------------------------------

        open_act = QtWidgets.QAction('Open dm3 or npy...', self)
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self.open_file)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction(open_act)

        # ------------------------------

        self.setCentralWidget(self.holo_widget)

        self.move(250, 50)
        self.setWindowTitle('Holo window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def open_file(self):
        file_dialog = QtWidgets.QFileDialog()
        img_path = file_dialog.getOpenFileName()[0]
        if img_path == '':
            print('No images to read. Exiting...')
            return

        print('Reading file "{0}"'.format(img_path))
        img_type = 'amp' if self.holo_widget.amp_radio_button.isChecked() else 'phs'

        # dm3 file
        if img_path.endswith('.dm3'):
            img_name_match = re.search('(.+)/(.+).dm3$', img_path)
            img_name_text = img_name_match.group(2)

            new_img = open_dm3_file(img_path, img_type)
        # npy file
        elif img_path.endswith('.npy'):
            img_name_match = re.search('(.+)/(.+).npy$', img_path)
            img_name_text = img_name_match.group(2)

            new_img_arr = np.load(img_path)
            h, w = new_img_arr.shape

            new_img = imsup.ImageExp(h, w, imsup.Image.cmp['CAP'])
            if img_type == 'amp':
                new_img.LoadAmpData(new_img_arr)
            else:
                new_img.LoadPhsData(new_img_arr)
        else:
            print('Could not load the image. It must be in dm3 or npy format...')
            return

        # in the case of npy file the px_dim will be the same as for the last dm3 file opened
        new_img.name = img_name_text
        new_img = rescale_image_buffer_to_window(new_img, const.disp_dim)

        self.holo_widget.insert_img_after_curr(new_img)

# --------------------------------------------------------

class HolographyWidget(QtWidgets.QWidget):
    def __init__(self):
        super(HolographyWidget, self).__init__()
        file_dialog = QtWidgets.QFileDialog()
        image_path = file_dialog.getOpenFileName()[0]
        if image_path == '':
            print('No images to read. Exiting...')
            exit()
        image = LoadImageSeriesFromFirstFile(image_path)
        self.display = LabelExt(self, image)
        self.display.setFixedWidth(const.disp_dim)
        self.display.setFixedHeight(const.disp_dim)
        self.plot_widget = PlotWidget()
        self.preview_scroll = ImgScrollArea(image)
        self.backup_image = None
        self.changes_made = []
        self.shift = [0, 0]
        self.rot_angle = 0
        self.scale_factor = 1.0
        self.warp_points = []
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(350)

        self.curr_info_label = QtWidgets.QLabel('', self)
        self.update_curr_info_label()

        # ------------------------------
        # Navigation panel (1)
        # ------------------------------

        prev_button = QtWidgets.QPushButton('Prev', self)
        next_button = QtWidgets.QPushButton('Next', self)
        lswap_button = QtWidgets.QPushButton('L-Swap', self)
        rswap_button = QtWidgets.QPushButton('R-Swap', self)
        flip_button = QtWidgets.QPushButton('Flip', self)
        set_name_button = QtWidgets.QPushButton('Set name', self)
        reset_names_button = QtWidgets.QPushButton('Reset names', self)
        zoom_button = QtWidgets.QPushButton('Crop N ROIs', self)
        delete_button = QtWidgets.QPushButton('Delete', self)
        clear_button = QtWidgets.QPushButton('Clear', self)
        undo_button = QtWidgets.QPushButton('Undo', self)
        add_marker_at_xy_button = QtWidgets.QPushButton('Add marker', self)

        self.clear_prev_checkbox = QtWidgets.QCheckBox('Clear prev. images', self)
        self.clear_prev_checkbox.setChecked(False)

        self.name_input = QtWidgets.QLineEdit(self.display.image.name, self)
        self.n_to_zoom_input = QtWidgets.QLineEdit('1', self)

        marker_xy_label = QtWidgets.QLabel('Marker xy-coords:')
        self.marker_x_input = QtWidgets.QLineEdit('0', self)
        self.marker_y_input = QtWidgets.QLineEdit('0', self)

        hbox_name = QtWidgets.QHBoxLayout()
        hbox_name.addWidget(set_name_button)
        hbox_name.addWidget(self.name_input)

        hbox_zoom = QtWidgets.QHBoxLayout()
        hbox_zoom.addWidget(zoom_button)
        hbox_zoom.addWidget(self.n_to_zoom_input)

        prev_button.clicked.connect(self.go_to_prev_image)
        next_button.clicked.connect(self.go_to_next_image)
        lswap_button.clicked.connect(self.swap_left)
        rswap_button.clicked.connect(self.swap_right)
        flip_button.clicked.connect(self.flip_image_h)
        set_name_button.clicked.connect(self.set_image_name)
        reset_names_button.clicked.connect(self.reset_image_names)
        zoom_button.clicked.connect(self.zoom_n_fragments)
        delete_button.clicked.connect(self.delete_image)
        clear_button.clicked.connect(self.clear_image)
        undo_button.clicked.connect(self.remove_last_point)
        add_marker_at_xy_button.clicked.connect(self.add_marker_at_xy)

        self.tab_nav = QtWidgets.QWidget()
        self.tab_nav.layout = QtWidgets.QGridLayout()
        self.tab_nav.layout.setColumnStretch(0, 1)
        self.tab_nav.layout.setColumnStretch(1, 1)
        self.tab_nav.layout.setColumnStretch(2, 1)
        self.tab_nav.layout.setColumnStretch(3, 1)
        self.tab_nav.layout.setColumnStretch(4, 1)
        self.tab_nav.layout.setColumnStretch(5, 1)
        self.tab_nav.layout.setRowStretch(0, 1)
        self.tab_nav.layout.setRowStretch(8, 1)
        self.tab_nav.layout.addWidget(prev_button, 1, 1, 1, 2)
        self.tab_nav.layout.addWidget(next_button, 1, 3, 1, 2)
        self.tab_nav.layout.addWidget(lswap_button, 2, 1, 1, 2)
        self.tab_nav.layout.addWidget(rswap_button, 2, 3, 1, 2)
        self.tab_nav.layout.addWidget(flip_button, 3, 1, 1, 2)
        self.tab_nav.layout.addWidget(clear_button, 3, 3, 1, 2)
        self.tab_nav.layout.addWidget(zoom_button, 4, 1)
        self.tab_nav.layout.addWidget(self.n_to_zoom_input, 4, 2)
        self.tab_nav.layout.addWidget(delete_button, 4, 3, 1, 2)
        self.tab_nav.layout.addWidget(set_name_button, 5, 1)
        self.tab_nav.layout.addWidget(self.name_input, 5, 2)
        self.tab_nav.layout.addWidget(undo_button, 5, 3, 1, 2)
        self.tab_nav.layout.addWidget(reset_names_button, 6, 1, 1, 2)
        self.tab_nav.layout.addWidget(self.clear_prev_checkbox, 6, 3, 1, 2)
        self.tab_nav.layout.addWidget(marker_xy_label, 7, 1)
        self.tab_nav.layout.addWidget(self.marker_x_input, 7, 2)
        self.tab_nav.layout.addWidget(self.marker_y_input, 7, 3)
        self.tab_nav.layout.addWidget(add_marker_at_xy_button, 7, 4)
        self.tab_nav.setLayout(self.tab_nav.layout)

        # ------------------------------
        # Display panel (2)
        # ------------------------------

        unwrap_button = QtWidgets.QPushButton('Unwrap', self)
        wrap_button = QtWidgets.QPushButton('Wrap', self)
        export_button = QtWidgets.QPushButton('Export', self)
        export_all_button = QtWidgets.QPushButton('Export all', self)
        blank_area_button = QtWidgets.QPushButton('Blank area', self)
        norm_phase_button = QtWidgets.QPushButton('Norm. phase', self)

        self.show_lines_checkbox = QtWidgets.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtWidgets.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.log_scale_checkbox = QtWidgets.QCheckBox('Log scale', self)
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_display)

        self.amp_radio_button = QtWidgets.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtWidgets.QRadioButton('Phase', self)
        self.cos_phs_radio_button = QtWidgets.QRadioButton('Phase cosine', self)
        self.amp_radio_button.setChecked(True)

        amp_phs_group = QtWidgets.QButtonGroup(self)
        amp_phs_group.addButton(self.amp_radio_button)
        amp_phs_group.addButton(self.phs_radio_button)
        amp_phs_group.addButton(self.cos_phs_radio_button)

        self.gray_radio_button = QtWidgets.QRadioButton('Grayscale', self)
        self.color_radio_button = QtWidgets.QRadioButton('Color', self)
        self.gray_radio_button.setChecked(True)

        color_group = QtWidgets.QButtonGroup(self)
        color_group.addButton(self.gray_radio_button)
        color_group.addButton(self.color_radio_button)

        self.export_tiff_radio_button = QtWidgets.QRadioButton('TIFF image', self)
        self.export_bin_radio_button = QtWidgets.QRadioButton('Numpy array file', self)
        self.export_tiff_radio_button.setChecked(True)

        export_group = QtWidgets.QButtonGroup(self)
        export_group.addButton(self.export_tiff_radio_button)
        export_group.addButton(self.export_bin_radio_button)

        fname_label = QtWidgets.QLabel('File name', self)
        self.fname_input = QtWidgets.QLineEdit(self.display.image.name, self)

        unwrap_button.clicked.connect(self.unwrap_img_phase)
        wrap_button.clicked.connect(self.wrap_img_phase)
        export_button.clicked.connect(self.export_image)
        export_all_button.clicked.connect(self.export_all)
        blank_area_button.clicked.connect(self.blank_area)
        norm_phase_button.clicked.connect(self.norm_phase)
        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)
        self.cos_phs_radio_button.toggled.connect(self.update_display)
        self.gray_radio_button.toggled.connect(self.update_display)
        self.color_radio_button.toggled.connect(self.update_display)

        grid_disp = QtWidgets.QGridLayout()
        grid_disp.setColumnStretch(0, 1)
        grid_disp.setColumnStretch(5, 1)
        grid_disp.setRowStretch(0, 1)
        # grid_disp.setRowStretch(4, 1)
        grid_disp.addWidget(self.show_lines_checkbox, 1, 1)
        grid_disp.addWidget(self.show_labels_checkbox, 2, 1)
        grid_disp.addWidget(self.log_scale_checkbox, 3, 1)
        grid_disp.addWidget(self.amp_radio_button, 1, 2)
        grid_disp.addWidget(self.phs_radio_button, 2, 2)
        grid_disp.addWidget(self.cos_phs_radio_button, 3, 2)
        grid_disp.addWidget(unwrap_button, 1, 4)
        grid_disp.addWidget(wrap_button, 2, 4)
        grid_disp.addWidget(self.gray_radio_button, 1, 3)
        grid_disp.addWidget(self.color_radio_button, 2, 3)
        grid_disp.addWidget(blank_area_button, 3, 3)
        grid_disp.addWidget(norm_phase_button, 3, 4)

        grid_exp = QtWidgets.QGridLayout()
        grid_exp.setColumnStretch(0, 1)
        grid_exp.setColumnStretch(4, 1)
        grid_exp.setRowStretch(0, 1)
        grid_exp.setRowStretch(3, 1)
        grid_exp.addWidget(fname_label, 1, 1)
        grid_exp.addWidget(self.fname_input, 2, 1)
        grid_exp.addWidget(export_button, 1, 2)
        grid_exp.addWidget(export_all_button, 2, 2)
        grid_exp.addWidget(self.export_tiff_radio_button, 1, 3)
        grid_exp.addWidget(self.export_bin_radio_button, 2, 3)

        self.tab_disp = QtWidgets.QWidget()
        self.tab_disp.layout = QtWidgets.QVBoxLayout()
        self.tab_disp.layout.addLayout(grid_disp)
        self.tab_disp.layout.addLayout(grid_exp)
        self.tab_disp.setLayout(self.tab_disp.layout)

        # ------------------------------
        # Manual alignment panel (3)
        # ------------------------------

        self.left_button = QtWidgets.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        self.right_button = QtWidgets.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        self.up_button = QtWidgets.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        self.down_button = QtWidgets.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        self.rot_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_right.png'), '', self)
        self.rot_counter_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_left.png'), '', self)
        self.apply_button = QtWidgets.QPushButton('Apply changes', self)
        self.reset_button = QtWidgets.QPushButton('Reset', self)

        self.manual_mode_checkbox = QtWidgets.QCheckBox('Manual mode', self)
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.clicked.connect(self.create_backup_image)

        self.px_shift_input = QtWidgets.QLineEdit('0', self)
        self.rot_angle_input = QtWidgets.QLineEdit('0.0', self)

        self.left_button.clicked.connect(self.move_left)
        self.right_button.clicked.connect(self.move_right)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)
        self.rot_clockwise_button.clicked.connect(self.rot_right)
        self.rot_counter_clockwise_button.clicked.connect(self.rot_left)
        self.apply_button.clicked.connect(self.apply_changes)
        self.reset_button.clicked.connect(self.reset_changes)

        self.disable_manual_panel()

        # ------------------------------
        # Automatic alignment panel (4)
        # ------------------------------

        auto_shift_button = QtWidgets.QPushButton('Auto-shift', self)
        auto_rot_button = QtWidgets.QPushButton('Auto-rotate', self)
        warpButton = QtWidgets.QPushButton('Warp', self)
        get_scale_ratio_button = QtWidgets.QPushButton('Get scale ratio from image calib.')
        scale_button = QtWidgets.QPushButton('Scale', self)
        reshift_button = QtWidgets.QPushButton('Re-Shift', self)
        rerot_button = QtWidgets.QPushButton('Re-Rot', self)
        rescale_button = QtWidgets.QPushButton('Re-Scale', self)
        rewarp_button = QtWidgets.QPushButton('Re-Warp', self)
        cross_corr_w_prev_button = QtWidgets.QPushButton('Cross corr. w. prev.', self)
        cross_corr_all_button = QtWidgets.QPushButton('Cross corr. all', self)

        self.scale_factor_input = QtWidgets.QLineEdit('1.0', self)

        auto_shift_button.clicked.connect(self.auto_shift_image)
        auto_rot_button.clicked.connect(self.auto_rot_image)
        warpButton.clicked.connect(partial(self.warp_image, False))
        get_scale_ratio_button.clicked.connect(self.get_scale_ratio_from_images)
        scale_button.clicked.connect(self.scale_image)
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
        grid_manual.addWidget(self.left_button, 2, 1)
        grid_manual.addWidget(self.right_button, 2, 3)
        grid_manual.addWidget(self.up_button, 1, 2)
        grid_manual.addWidget(self.down_button, 3, 2)
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
        grid_auto.addWidget(warpButton, 5, 1)
        grid_auto.addWidget(reshift_button, 1, 2)
        grid_auto.addWidget(rerot_button, 2, 2)
        grid_auto.addWidget(rescale_button, 4, 2)
        grid_auto.addWidget(rewarp_button, 5, 2)
        grid_auto.addWidget(cross_corr_w_prev_button, 1, 3)
        grid_auto.addWidget(cross_corr_all_button, 2, 3)
        grid_auto.addWidget(self.scale_factor_input, 4, 3)

        self.tab_align = QtWidgets.QWidget()
        self.tab_align.layout = QtWidgets.QVBoxLayout()
        self.tab_align.layout.addLayout(grid_manual)
        self.tab_align.layout.addLayout(grid_auto)
        self.tab_align.setLayout(self.tab_align.layout)

        # ------------------------------
        # Holography panel (5)
        # ------------------------------

        holo_no_ref_1_button = QtWidgets.QPushButton('FFT', self)
        holo_no_ref_2_button = QtWidgets.QPushButton('Holo', self)
        holo_with_ref_2_button = QtWidgets.QPushButton('Holo+Ref', self)
        holo_no_ref_3_button = QtWidgets.QPushButton('IFFT', self)
        sum_button = QtWidgets.QPushButton('Sum', self)
        diff_button = QtWidgets.QPushButton('Diff', self)
        amplify_button = QtWidgets.QPushButton('Amplify', self)
        add_radians_button = QtWidgets.QPushButton('Add radians', self)
        remove_phase_tilt_button = QtWidgets.QPushButton('Remove phase tilt', self)
        get_sideband_from_xy_button = QtWidgets.QPushButton('Get sideband', self)
        do_all_button = QtWidgets.QPushButton('DO ALL', self)

        self.subpixel_shift_checkbox = QtWidgets.QCheckBox('Subpixel shift', self)
        self.subpixel_shift_checkbox.setChecked(False)

        aperture_label = QtWidgets.QLabel('Aperture rad. [px]', self)
        self.aperture_input = QtWidgets.QLineEdit(str(const.aperture), self)

        hann_win_label = QtWidgets.QLabel('Hann window [px]', self)
        self.hann_win_input = QtWidgets.QLineEdit(str(const.hann_win), self)

        amp_factor_label = QtWidgets.QLabel('Amp. factor', self)
        self.amp_factor_input = QtWidgets.QLineEdit('2.0', self)

        self.radians2add_input = QtWidgets.QLineEdit('3.14', self)

        sideband_xy_label = QtWidgets.QLabel('Sideband xy-coords:')
        self.sideband_x_input = QtWidgets.QLineEdit('0', self)
        self.sideband_y_input = QtWidgets.QLineEdit('0', self)

        hbox_holo = QtWidgets.QHBoxLayout()
        hbox_holo.addWidget(holo_no_ref_2_button)
        hbox_holo.addWidget(holo_with_ref_2_button)

        holo_no_ref_1_button.clicked.connect(self.rec_holo_no_ref_1)
        holo_no_ref_2_button.clicked.connect(self.rec_holo_no_ref_2)
        holo_with_ref_2_button.clicked.connect(self.rec_holo_with_ref_2)
        holo_no_ref_3_button.clicked.connect(self.rec_holo_no_ref_3)
        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)
        amplify_button.clicked.connect(self.amplify_phase)
        add_radians_button.clicked.connect(self.add_radians)
        remove_phase_tilt_button.clicked.connect(self.remove_phase_tilt)
        get_sideband_from_xy_button.clicked.connect(self.get_sideband_from_xy)
        do_all_button.clicked.connect(self.do_all)

        self.tab_holo = QtWidgets.QWidget()
        self.tab_holo.layout = QtWidgets.QGridLayout()
        self.tab_holo.layout.setColumnStretch(0, 1)
        self.tab_holo.layout.setColumnStretch(1, 1)
        self.tab_holo.layout.setColumnStretch(2, 1)
        self.tab_holo.layout.setColumnStretch(3, 1)
        self.tab_holo.layout.setColumnStretch(4, 1)
        self.tab_holo.layout.setColumnStretch(5, 1)
        self.tab_holo.layout.setColumnStretch(6, 1)
        self.tab_holo.layout.setColumnStretch(7, 1)
        self.tab_holo.layout.setRowStretch(0, 1)
        self.tab_holo.layout.setRowStretch(8, 1)
        self.tab_holo.layout.addWidget(aperture_label, 1, 1, 1, 2)
        self.tab_holo.layout.addWidget(self.aperture_input, 2, 1, 1, 2)
        self.tab_holo.layout.addWidget(hann_win_label, 1, 3, 1, 2)
        self.tab_holo.layout.addWidget(self.hann_win_input, 2, 3, 1, 2)
        self.tab_holo.layout.addWidget(holo_no_ref_1_button, 3, 1, 1, 2)
        self.tab_holo.layout.addLayout(hbox_holo, 3, 3, 1, 2)
        self.tab_holo.layout.addWidget(holo_no_ref_3_button, 4, 1, 1, 2)
        self.tab_holo.layout.addWidget(sum_button, 4, 3, 1, 2)
        self.tab_holo.layout.addWidget(diff_button, 4, 5, 1, 2)
        self.tab_holo.layout.addWidget(amp_factor_label, 1, 5, 1, 2)
        self.tab_holo.layout.addWidget(self.amp_factor_input, 2, 5, 1, 2)
        self.tab_holo.layout.addWidget(amplify_button, 3, 5, 1, 2)
        self.tab_holo.layout.addWidget(add_radians_button, 5, 1, 1, 1)
        self.tab_holo.layout.addWidget(self.radians2add_input, 5, 2, 1, 1)
        self.tab_holo.layout.addWidget(sideband_xy_label, 5, 3, 1, 2)
        self.tab_holo.layout.addWidget(self.sideband_x_input, 5, 5, 1, 2)
        self.tab_holo.layout.addWidget(self.sideband_y_input, 6, 5, 1, 2)
        self.tab_holo.layout.addWidget(get_sideband_from_xy_button, 6, 3, 1, 2)
        self.tab_holo.layout.addWidget(do_all_button, 6, 1, 1, 2)
        self.tab_holo.layout.addWidget(self.subpixel_shift_checkbox, 7, 1, 1, 2)
        self.tab_holo.layout.addWidget(remove_phase_tilt_button, 7, 3, 1, 2)
        self.tab_holo.setLayout(self.tab_holo.layout)

        # ------------------------------
        # Magnetic calculations panel (6)
        # ------------------------------

        plot_button = QtWidgets.QPushButton('Plot profile', self)
        calc_B_sec_button = QtWidgets.QPushButton('Calc. B from section', self)
        calc_B_prof_button = QtWidgets.QPushButton('Calc. B from profile')
        calc_grad_button = QtWidgets.QPushButton('Calculate gradient', self)
        calc_Bxy_maps_button = QtWidgets.QPushButton('Calc. Bx, By maps', self)
        calc_B_polar_button = QtWidgets.QPushButton('Calc. B polar', self)
        gen_B_stats_button = QtWidgets.QPushButton('Gen. B statistics', self)
        calc_MIP_button = QtWidgets.QPushButton('Calc. MIP', self)
        filter_contours_button = QtWidgets.QPushButton('Filter contours', self)
        export_glob_scaled_phases_button = QtWidgets.QPushButton('Export phase colmaps', self)
        export_img3d_button = QtWidgets.QPushButton('Export 3D image', self)

        self.add_arrows_checkbox = QtWidgets.QCheckBox('Add grad. arrows', self)
        self.add_arrows_checkbox.setChecked(False)

        self.perpendicular_arrows_checkbox = QtWidgets.QCheckBox('Perpendicular', self)
        self.perpendicular_arrows_checkbox.setChecked(False)

        self.orig_in_pt1_radio_button = QtWidgets.QRadioButton('Orig in pt1', self)
        self.orig_in_mid_radio_button = QtWidgets.QRadioButton('Orig in middle', self)
        self.orig_in_pt1_radio_button.setChecked(True)

        orig_B_polar_group = QtWidgets.QButtonGroup(self)
        orig_B_polar_group.addButton(self.orig_in_pt1_radio_button)
        orig_B_polar_group.addButton(self.orig_in_mid_radio_button)

        int_width_label = QtWidgets.QLabel('Profile width [px]', self)
        self.int_width_input = QtWidgets.QLineEdit('1', self)

        sample_thick_label = QtWidgets.QLabel('Sample thickness [nm]', self)
        self.sample_thick_input = QtWidgets.QLineEdit('30', self)

        threshold_label = QtWidgets.QLabel('Int. threshold [0-1]', self)
        self.threshold_input = QtWidgets.QLineEdit('0.9', self)

        num_of_r_iters_label = QtWidgets.QLabel('# R iters', self)
        self.num_of_r_iters_input = QtWidgets.QLineEdit('1', self)

        arr_size_label = QtWidgets.QLabel('Arrow size', self)
        arr_dist_label = QtWidgets.QLabel('Arrow dist.', self)
        self.arr_size_input = QtWidgets.QLineEdit('20', self)
        self.arr_dist_input = QtWidgets.QLineEdit('50', self)

        self.only_int = QtGui.QIntValidator()
        self.arr_size_input.setValidator(self.only_int)
        self.arr_dist_input.setValidator(self.only_int)

        ph3d_ang1_label = QtWidgets.QLabel('Ang #1', self)
        ph3d_ang2_label = QtWidgets.QLabel('Ang #2', self)
        self.ph3d_ang1_input = QtWidgets.QLineEdit('0', self)
        self.ph3d_ang2_input = QtWidgets.QLineEdit('0', self)

        ph3d_mesh_label = QtWidgets.QLabel('Mesh dist.', self)
        self.ph3d_mesh_input = QtWidgets.QLineEdit('50', self)
        self.ph3d_mesh_input.setValidator(self.only_int)

        acc_voltage_label = QtWidgets.QLabel('U_acc [kV]', self)
        self.acc_voltage_input = QtWidgets.QLineEdit('300', self)

        arr_size_vbox = QtWidgets.QVBoxLayout()
        arr_size_vbox.addWidget(arr_size_label)
        arr_size_vbox.addWidget(self.arr_size_input)

        arr_dist_vbox = QtWidgets.QVBoxLayout()
        arr_dist_vbox.addWidget(arr_dist_label)
        arr_dist_vbox.addWidget(self.arr_dist_input)

        ph3d_ang1_vbox = QtWidgets.QVBoxLayout()
        ph3d_ang1_vbox.addWidget(ph3d_ang1_label)
        ph3d_ang1_vbox.addWidget(self.ph3d_ang1_input)

        ph3d_ang2_vbox = QtWidgets.QVBoxLayout()
        ph3d_ang2_vbox.addWidget(ph3d_ang2_label)
        ph3d_ang2_vbox.addWidget(self.ph3d_ang2_input)

        plot_button.clicked.connect(self.plot_profile)
        calc_B_sec_button.clicked.connect(self.calc_B_from_section)
        calc_B_prof_button.clicked.connect(self.calc_B_from_profile)
        calc_grad_button.clicked.connect(self.calc_phase_gradient)
        calc_Bxy_maps_button.clicked.connect(self.calc_Bxy_maps)
        calc_B_polar_button.clicked.connect(self.calc_B_polar_from_section)
        gen_B_stats_button.clicked.connect(self.gen_phase_stats)
        calc_MIP_button.clicked.connect(self.calc_mean_inner_potential)
        filter_contours_button.clicked.connect(self.filter_contours)
        export_glob_scaled_phases_button.clicked.connect(self.export_glob_sc_phases)
        export_img3d_button.clicked.connect(self.export_3d_image)

        self.tab_calc = QtWidgets.QWidget()
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
        self.tab_calc.layout.setRowStretch(11, 1)
        self.tab_calc.layout.addWidget(sample_thick_label, 1, 1, 1, 2)
        self.tab_calc.layout.addWidget(self.sample_thick_input, 2, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_grad_button, 3, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_sec_button, 4, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_prof_button, 5, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_Bxy_maps_button, 6, 1, 1, 2)
        self.tab_calc.layout.addWidget(calc_B_polar_button, 7, 1, 1, 2)
        self.tab_calc.layout.addWidget(num_of_r_iters_label, 8, 1, 1, 1)
        self.tab_calc.layout.addWidget(self.num_of_r_iters_input, 8, 2, 1, 1)
        self.tab_calc.layout.addWidget(self.orig_in_pt1_radio_button, 9, 1, 1, 2)
        self.tab_calc.layout.addWidget(self.orig_in_mid_radio_button, 10, 1, 1, 2)
        self.tab_calc.layout.addWidget(int_width_label, 1, 3, 1, 2)
        self.tab_calc.layout.addWidget(self.int_width_input, 2, 3, 1, 2)
        self.tab_calc.layout.addWidget(plot_button, 3, 3, 1, 2)
        self.tab_calc.layout.addLayout(arr_size_vbox, 4, 3, 2, 1)
        self.tab_calc.layout.addLayout(arr_dist_vbox, 4, 4, 2, 1)
        self.tab_calc.layout.addWidget(export_glob_scaled_phases_button, 6, 3, 1, 2)
        self.tab_calc.layout.addWidget(self.add_arrows_checkbox, 7, 3, 1, 2)
        self.tab_calc.layout.addWidget(self.perpendicular_arrows_checkbox, 8, 3, 1, 2)
        self.tab_calc.layout.addWidget(gen_B_stats_button, 9, 3, 1, 2)
        self.tab_calc.layout.addWidget(threshold_label, 1, 5, 1, 2)
        self.tab_calc.layout.addWidget(self.threshold_input, 2, 5, 1, 2)
        self.tab_calc.layout.addWidget(filter_contours_button, 3, 5, 1, 2)
        self.tab_calc.layout.addLayout(ph3d_ang1_vbox, 4, 5, 2, 1)
        self.tab_calc.layout.addLayout(ph3d_ang2_vbox, 4, 6, 2, 1)
        self.tab_calc.layout.addWidget(ph3d_mesh_label, 6, 5)
        self.tab_calc.layout.addWidget(self.ph3d_mesh_input, 6, 6)
        self.tab_calc.layout.addWidget(export_img3d_button, 7, 5, 1, 2)
        self.tab_calc.layout.addWidget(acc_voltage_label, 8, 5)
        self.tab_calc.layout.addWidget(self.acc_voltage_input, 8, 6)
        self.tab_calc.layout.addWidget(calc_MIP_button, 9, 5, 1, 2)
        self.tab_calc.setLayout(self.tab_calc.layout)

        # ------------------------------
        # Bright/Gamma/Contrast panel (7)
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

        self.tab_corr = QtWidgets.QWidget()
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

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_nav, 'Navigation')
        self.tabs.addTab(self.tab_disp, 'Display')
        self.tabs.addTab(self.tab_align, 'Alignment')
        self.tabs.addTab(self.tab_holo, 'Holography')
        self.tabs.addTab(self.tab_calc, 'Magnetic field')
        self.tabs.addTab(self.tab_corr, 'Corrections')

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
        curr_img = self.display.image
        disp_name = curr_img.name[:const.disp_name_max_len]
        if len(curr_img.name) > const.disp_name_max_len:
            disp_name = disp_name[:-3] + '...'
        self.curr_info_label.setText('{0}, dim = {1} px'.format(disp_name, curr_img.width))

    def enable_manual_panel(self):
        self.left_button.setEnabled(True)
        self.right_button.setEnabled(True)
        self.up_button.setEnabled(True)
        self.down_button.setEnabled(True)
        self.rot_clockwise_button.setEnabled(True)
        self.rot_counter_clockwise_button.setEnabled(True)
        self.px_shift_input.setEnabled(True)
        self.rot_angle_input.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    def disable_manual_panel(self):
        if self.backup_image is not None:
            self.reset_changes()
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)
        self.up_button.setEnabled(False)
        self.down_button.setEnabled(False)
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
        first_img = imsup.GetFirstImage(curr_img)
        img_queue = imsup.CreateImageListFromFirstImage(first_img)
        for img, idx in zip(img_queue, range(len(img_queue))):
            img.numInSeries = idx + 1
            img.name = 'img0{0}'.format(idx + 1) if idx < 9 else 'img{0}'.format(idx + 1)
        self.update_curr_info_label()
        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)

    def go_to_image(self, new_idx):
        first_img = imsup.GetFirstImage(self.display.image)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        if new_idx > len(imgs) - 1:
            new_idx = len(imgs) - 1
        curr_img = imgs[new_idx]
        if curr_img.name == '':
            curr_img.name = 'img0{0}'.format(new_idx + 1) if new_idx < 9 else 'img{0}'.format(new_idx + 1)
        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)
        self.manual_mode_checkbox.setChecked(False)
        self.disable_manual_panel()
        self.display.image = imgs[new_idx]
        if len(self.display.pointSets) < self.display.image.numInSeries:
            self.display.pointSets.append([])
        self.display.update_labels()
        self.update_curr_info_label()
        self.update_display_and_bcg()

    def go_to_prev_image(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            return
        prev_idx = curr_img.prev.numInSeries - 1
        self.go_to_image(prev_idx)

    def go_to_next_image(self):
        curr_img = self.display.image
        if curr_img.next is None:
            return
        next_idx = curr_img.next.numInSeries - 1
        self.go_to_image(next_idx)

    def go_to_last_image(self):
        curr_img = self.display.image
        last_img = imsup.GetLastImage(curr_img)
        last_idx = last_img.numInSeries - 1
        self.go_to_image(last_idx)

    def flip_image_h(self):
        curr = self.display.image
        imsup.flip_image_h(curr)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        self.display.setImage()

    def save_amp_as_tiff(self, fname, log, color):
        curr_img = self.display.image
        amp = np.copy(curr_img.amPh.am)
        if log:
            amp[np.where(amp <= 0)] = 1e-5
            amp = np.log(amp)
        amp = update_image_bright_cont_gamma(amp, brg=curr_img.bias, cnt=curr_img.gain, gam=curr_img.gamma)
        if color:
            amp = imsup.grayscale_to_rgb(amp)
            amp_to_save = imsup.im.fromarray(amp.astype(np.uint8), 'RGB')
        else:
            amp_to_save = imsup.im.fromarray(amp.astype(np.uint8))
        amp_to_save.save('{0}.tif'.format(fname))

    def save_phs_as_tiff(self, fname, log, color):
        curr_img = self.display.image
        phs = np.copy(curr_img.amPh.ph)
        if log:
            phs[np.where(phs <= 0)] = 1e-5
            phs = np.log(phs)
        phs = update_image_bright_cont_gamma(phs, brg=curr_img.bias, cnt=curr_img.gain, gam=curr_img.gamma)
        if color:
            phs = imsup.grayscale_to_rgb(phs)
            phs_to_save = imsup.im.fromarray(phs.astype(np.uint8), 'RGB')
        else:
            phs_to_save = imsup.im.fromarray(phs.astype(np.uint8))
        phs_to_save.save('{0}.tif'.format(fname))

    def export_image(self):
        curr_num = self.display.image.numInSeries
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

        # binary file (deprecated)
        # if self.export_bin_radio_button.isChecked():
        #     fname_ext = ''
        #     if is_amp_checked:
        #         curr_img.amPh.am.tofile(fname)
        #     elif is_phs_checked:
        #         curr_img.amPh.ph.tofile(fname)
        #     else:
        #         cos_phs = np.cos(curr_img.amPh.ph)
        #         cos_phs.tofile(fname)
        #     print('Saved image to binary file: "{0}"'.format(fname))

        # numpy array file (new)
        if self.export_bin_radio_button.isChecked():
            fname_ext = '.npy'

            if is_amp_checked:
                np.save(fname, curr_img.amPh.am)
            elif is_phs_checked:
                np.save(fname, curr_img.amPh.ph)
            else:
                cos_phs = np.cos(curr_img.amPh.ph)
                np.save(fname, cos_phs)
            print('Saved image to numpy array file: "{0}.npy"'.format(fname))
        # TIF file
        else:
            fname_ext = '.tif'
            log = True if self.log_scale_checkbox.isChecked() else False
            color = True if self.color_radio_button.isChecked() else False

            if is_amp_checked:
                self.save_amp_as_tiff(fname, log, color)
            elif is_phs_checked:
                self.save_phs_as_tiff(fname, log, color)
            else:
                phs_tmp = np.copy(curr_img.amPh.ph)
                curr_img.amPh.ph = np.cos(phs_tmp)
                self.save_phs_as_tiff(fname, log, color)
                curr_img.amPh.ph = np.copy(phs_tmp)
            print('Saved image as "{0}.tif"'.format(fname))

        # save log file
        log_fname = '{0}_log.txt'.format(fname)
        with open(log_fname, 'w') as log_file:
            log_file.write('File name:\t{0}{1}\n'
                           'Image name:\t{2}\n'
                           'Image size:\t{3}x{4}\n'
                           'Data type:\t{5}\n'
                           'Calibration:\t{6} nm\n'.format(fname, fname_ext, curr_img.name, curr_img.width,
                                                           curr_img.height, curr_img.amPh.am.dtype,
                                                           curr_img.px_dim * 1e9))
        print('Saved log file: "{0}"'.format(log_fname))

    def export_all(self):
        curr_img = imsup.GetFirstImage(self.display.image)
        self.display.image = curr_img
        self.export_image()
        while curr_img.next is not None:
            self.go_to_next_image()
            self.export_image()
            curr_img = self.display.image
        print('All images saved')

    def delete_image(self):
        curr_img = self.display.image
        if curr_img.prev is None and curr_img.next is None:
            return

        curr_idx = curr_img.numInSeries - 1
        first_img = imsup.GetFirstImage(curr_img)
        all_img_list = imsup.CreateImageListFromFirstImage(first_img)

        new_idx = curr_idx - 1 if curr_img.prev is not None else curr_idx + 1
        self.go_to_image(new_idx)

        del all_img_list[curr_idx]
        del self.display.pointSets[curr_idx]

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
        self.update_display()

    def update_display(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.setImage(dispAmp=is_amp_checked, dispPhs=is_phs_checked,
                              logScale=is_log_scale_checked, color=is_color_checked)

    def update_bcg(self):
        bright_val = int(self.bright_input.text())
        cont_val = int(self.cont_input.text())
        gamma_val = float(self.gamma_input.text())

        self.change_bright_slider_value()
        self.change_cont_slider_value()
        self.change_gamma_slider_value()

        self.display.setImage(update_bcg=True, bright=bright_val, cont=cont_val, gamma=gamma_val)

    def update_display_and_bcg(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()

        bright_val = int(self.bright_input.text())
        cont_val = int(self.cont_input.text())
        gamma_val = float(self.gamma_input.text())

        self.change_bright_slider_value()
        self.change_cont_slider_value()
        self.change_gamma_slider_value()

        self.display.setImage(dispAmp=is_amp_checked, dispPhs=is_phs_checked,
                              logScale=is_log_scale_checked, color=is_color_checked,
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

    def unwrap_img_phase(self):
        curr_img = self.display.image
        new_phs = tr.unwrap_phase(curr_img.amPh.ph)
        curr_img.amPh.ph = np.copy(new_phs)
        self.display.image = rescale_image_buffer_to_window(curr_img, const.disp_dim)
        self.update_display()

    def wrap_img_phase(self):
        curr_img = self.display.image
        uw_min = np.min(curr_img.amPh.ph)

        if uw_min > 0:
            uw_min = 0
        new_phs = (curr_img.amPh.ph - uw_min) % (2 * np.pi) - np.pi

        curr_img.amPh.ph = np.copy(new_phs)
        self.display.image = rescale_image_buffer_to_window(curr_img, const.disp_dim)
        self.update_display()

    def blank_area(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        p1, p2 = self.display.pointSets[curr_idx][:2]
        p1 = CalcRealTLCoords(curr_img.width, p1)
        p2 = CalcRealTLCoords(curr_img.width, p2)

        blanked_img = imsup.copy_am_ph_image(curr_img)
        blanked_img.amPh.am[p1[1]:p2[1], p1[0]:p2[0]] = 0.0
        blanked_img.amPh.ph[p1[1]:p2[1], p1[0]:p2[0]] = 0.0

        blanked_img.name = '{0}_b'.format(curr_img.name)
        self.insert_img_after_curr(blanked_img)

    def norm_phase(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        n_points = len(self.display.pointSets[curr_idx])
        if n_points == 0:
            print('Mark reference point (or area) on the image')
            return

        pt_disp = self.display.pointSets[curr_idx][:n_points]
        pt_real = CalcRealTLCoordsForSetOfPoints(curr_img.width, pt_disp)

        x1, y1 = pt_real[0]
        x2, y2 = 0, 0
        if n_points > 1:
            x2, y2 = pt_real[1]

        first_img = imsup.GetFirstImage(curr_img)
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        for img in img_list:
            if n_points == 1:
                norm_val = img.amPh.ph[y1, x1]
            else:
                norm_val = np.average(img.amPh.ph[y1:y2, x1:x2])
            img.amPh.ph -= norm_val
            img.update_cos_phase()
            img.name += '_' if norm_val > 0.0 else '_+'
            img.name += '{0:.2f}rad'.format(-norm_val)

        self.update_display()
        self.update_curr_info_label()
        print('All phases normalized')

    def export_3d_image(self):
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

        curr_img = self.display.image
        img_dim = curr_img.width
        px_sz = curr_img.px_dim * 1e9

        ang1 = int(self.ph3d_ang1_input.text())
        ang2 = int(self.ph3d_ang2_input.text())
        step = int(self.ph3d_mesh_input.text())

        # X = np.arange(0, img_dim, step, dtype=np.float32)
        # Y = np.arange(0, img_dim, step, dtype=np.float32)
        # phs_to_disp = np.copy(curr_img.amPh.ph[0:img_dim:step, 0:img_dim:step])

        X = np.arange(0, img_dim, dtype=np.float32)
        Y = np.arange(0, img_dim, dtype=np.float32)
        X, Y = np.meshgrid(X, Y)
        X *= px_sz
        Y *= px_sz

        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, curr_img.amPh.ph, cmap=cm.jet, rstride=step, cstride=step)    # mesh step (dist. between rows/cols used)
        # ax.plot_surface(X, Y, curr_img.amPh.ph, cmap=cm.jet, rcount=step, ccount=step)    # mesh (how many rows, cols will be used)
        # ax.plot_surface(X, Y, phs_to_disp, cmap=cm.jet)

        ax.view_init(ang1, ang2)
        plt.savefig('{0}_{1}_{2}.png'.format(curr_img.name, ang1, ang2), dpi=300)
        print('3D image exported!')
        plt.clf()
        plt.cla()
        plt.close(fig)

    def export_glob_sc_phases(self):
        first_img = imsup.GetFirstImage(self.display.image)
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        is_arrows_checked = self.add_arrows_checkbox.isChecked()
        is_perpendicular_checked = self.perpendicular_arrows_checkbox.isChecked()
        arrow_size = int(self.arr_size_input.text())
        arrow_dist = int(self.arr_dist_input.text())
        export_glob_sc_images(img_list, is_arrows_checked, is_perpendicular_checked, arrow_size, arrow_dist, cbar_lab='phase shift [rad]')
        print('Phases exported!')

    def zoom_n_fragments(self):
        curr_idx = self.display.image.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) < 2:
            print('You have to mark two points on the image in order to zoom!')
            return

        curr_img = self.display.image
        pt1, pt2 = self.display.pointSets[curr_idx][:2]
        pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
        disp_crop_coords = pt1 + pt2
        real_tl_coords = CalcRealTLCoords(curr_img.width, disp_crop_coords)
        real_sq_coords = imsup.MakeSquareCoords(real_tl_coords)
        if np.abs(real_sq_coords[2] - real_sq_coords[0]) % 2:
            real_sq_coords[2] += 1
            real_sq_coords[3] += 1

        print(real_sq_coords)

        n_to_zoom = np.int(self.n_to_zoom_input.text())
        first_img = imsup.GetFirstImage(curr_img)
        insert_idx = curr_idx + n_to_zoom
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        img_list2 = img_list[curr_idx:insert_idx]

        for img, n in zip(img_list2, range(insert_idx, insert_idx + n_to_zoom)):
            frag = zoom_fragment(img, real_sq_coords)
            frag.name = 'crop_from_{0}'.format(img.name)
            print(frag.width, frag.height)
            img_list.insert(n, frag)
            self.display.pointSets.insert(n, [])

        img_list.UpdateLinks()

        if self.clear_prev_checkbox.isChecked():
            del img_list[curr_idx:insert_idx]
            del self.display.pointSets[curr_idx:insert_idx]

        self.go_to_image(curr_idx)
        print('Zooming complete!')

    def clear_image(self):
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
        self.display.pointSets[self.display.image.numInSeries - 1][:] = []
        self.display.repaint()

    def remove_last_point(self):
        curr_idx = self.display.image.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) == 0:
            return
        all_labels = self.display.children()
        if len(all_labels) > 0:
            last_label = all_labels[-1]
            last_label.deleteLater()
        del self.display.pointSets[curr_idx][-1]
        self.display.repaint()

    def add_marker_at_xy(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        curr_pos = [ int(self.marker_x_input.text()), int(self.marker_y_input.text()) ]
        if 0 <= curr_pos[0] < const.disp_dim and 0 <= curr_pos[1] < const.disp_dim:
            # --- to be removed later ---
            # for idx in range(len(self.display.pointSets)):
            #     self.display.pointSets[idx].append(curr_pos)
            # -----------------------
            self.display.pointSets[curr_idx].append(curr_pos)         # uncomment later
            self.display.repaint()
            if self.display.show_labs:
                self.display.show_last_label()

            pt_idx = len(self.display.pointSets[curr_idx])
            disp_x, disp_y = curr_pos
            real_x, real_y = CalcRealTLCoords(curr_img.width, curr_pos)
            print('Added point {0} at:\nx = {1}\ny = {2}'.format(pt_idx, disp_x, disp_y))
            print('Actual position:\nx = {0}\ny = {1}'.format(real_x, real_y))
            print('Amp = {0:.2f}\nPhs = {1:.2f}'.format(curr_img.amPh.am[real_y, real_x], curr_img.amPh.ph[real_y, real_x]))
        else:
            print('Wrong marker coordinates!')

    def create_backup_image(self):
        if self.manual_mode_checkbox.isChecked():
            self.backup_image = imsup.copy_am_ph_image(self.display.image)
            self.enable_manual_panel()
        else:
            self.backup_image = None
            self.disable_manual_panel()

    def move_left(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([0, -n_px])

    def move_right(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([0, n_px])

    def move_up(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([-n_px, 0])

    def move_down(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([n_px, 0])

    def move_image(self, shift):
        bckp = self.backup_image
        curr = self.display.image
        total_shift = list(np.array(curr.shift) + np.array(shift))

        if curr.rot != 0:
            tmp = tr.RotateImageSki(bckp, curr.rot)
            shifted_img = imsup.shift_am_ph_image(tmp, total_shift)
        else:
            shifted_img = imsup.shift_am_ph_image(bckp, total_shift)

        curr.amPh.am = np.copy(shifted_img.amPh.am)
        curr.amPh.ph = np.copy(shifted_img.amPh.ph)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        self.display.image.shift = total_shift
        # self.display.setImage()
        self.update_display_and_bcg()

    def rot_left(self):
        ang = float(self.rot_angle_input.text())
        self.rotate_image(ang)

    def rot_right(self):
        ang = float(self.rot_angle_input.text())
        self.rotate_image(-ang)

    def rotate_image(self, rot):
        bckp = self.backup_image
        curr = self.display.image
        total_rot = curr.rot + rot

        if curr.shift != [0, 0]:
            tmp = imsup.shift_am_ph_image(bckp, curr.shift)
            rotated_img = tr.RotateImageSki(tmp, total_rot)
        else:
            rotated_img = tr.RotateImageSki(bckp, total_rot)

        curr.amPh.am = np.copy(rotated_img.amPh.am)
        curr.amPh.ph = np.copy(rotated_img.amPh.ph)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        curr.rot = total_rot
        # self.display.setImage()
        self.update_display_and_bcg()

    # def repeat_prev_mods(self):
    #     curr = imsup.copy_am_ph_image(self.backup_image)
    #     for mod in self.changes_made:
    #         curr = modify_image(curr, mod[:2], bool(mod[2]))
    #     self.display.image = curr

    def zero_shift_rot(self):
        self.display.image.shift = [0, 0]
        self.display.image.rot = 0

    def apply_changes(self):
        self.zero_shift_rot()
        self.backup_image = imsup.copy_am_ph_image(self.display.image)
        print('Changes for {0} have been applied'.format(self.display.image.name))

    def reset_changes(self):
        curr = self.display.image
        self.zero_shift_rot()
        curr.amPh.am = np.copy(self.backup_image.amPh.am)
        curr.amPh.ph = np.copy(self.backup_image.amPh.ph)
        self.display.image = rescale_image_buffer_to_window(curr, const.disp_dim)
        self.backup_image = None

    def reset_changes_and_update_display(self):
        self.reset_changes()
        self.display.setImage()
        print('Changes for {0} have been revoked'.format(self.display.image.name))

    def cross_corr_with_prev(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            print('There is no reference image!')
            return
        img_list_to_cc = imsup.CreateImageListFromImage(curr_img.prev, 2)
        img_aligned = cross_corr_images(img_list_to_cc)[0]
        self.insert_img_after_curr(img_aligned)
        self.go_to_next_image()

    def cross_corr_all(self):
        curr_img = self.display.image
        first_img = imsup.GetFirstImage(curr_img)
        all_img_list = imsup.CreateImageListFromFirstImage(first_img)
        n_imgs = len(all_img_list)
        insert_idx = n_imgs
        img_align_list = cross_corr_images(all_img_list)

        ref_img = imsup.copy_am_ph_image(first_img)
        img_align_list.insert(0, ref_img)
        all_img_list += img_align_list
        for i in range(n_imgs):
            self.display.pointSets.append([])
        all_img_list.UpdateAndRestrainLinks()

        self.go_to_image(insert_idx)
        print('Cross-correlation done!')

    # def align_images(self):
    #     if self.shift_radio_button.isChecked():
    #         self.align_shift()
    #     else:
    #         self.align_rot_dev()

    def auto_rot_image(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1

        # p11, p12 = self.display.pointSets[curr_idx-1][:2]
        # p21, p22 = self.display.pointSets[curr_idx][:2]

        points1 = self.display.pointSets[curr_idx-1]
        points2 = self.display.pointSets[curr_idx]

        np1 = len(points1)
        np2 = len(points2)

        if np1 != np2:
            print('Mark the same number of points on both images!')
            return

        if np1 % 2:
            np1 -= 1
            np2 -= 1
            points1 = points1[:-1]
            points2 = points2[:-1]

        line1 = tr.Line(0, 0)
        line2 = tr.Line(0, 0)
        rot_angle_avg = 0.0
        n_pairs = np1 // 2

        for l in range(n_pairs):
            p11, p12 = points1[2*l:2*(l+1)]
            p21, p22 = points2[2*l:2*(l+1)]
            line1.getFromPoints(p11, p12)
            line2.getFromPoints(p21, p22)
            rot_angle = imsup.Degrees(np.arctan(line2.a) - np.arctan(line1.a))
            print('Rot. angle = {0:.2f} deg'.format(rot_angle))
            rot_angle_avg += rot_angle

        rot_angle_avg /= n_pairs
        self.rot_angle = rot_angle_avg
        print('Avg. rot. angle = {0:.2f} deg'.format(rot_angle_avg))

        img_rot = tr.RotateImageSki(curr_img, rot_angle_avg)

        img_rot.name = curr_img.name + '_rot'
        self.insert_img_after_curr(img_rot)

    def auto_shift_rot_old(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points1 = self.display.pointSets[curr_idx-1]
        points2 = self.display.pointSets[curr_idx]
        n_points1 = len(points1)
        n_points2 = len(points2)

        if n_points1 != n_points2:
            print('Mark the same number of points on both images!')
            return

        poly1 = [ CalcRealCoords(img_width, pt1) for pt1 in points1 ]
        poly2 = [ CalcRealCoords(img_width, pt2) for pt2 in points2 ]

        rcSum = [0, 0]
        rotCenters = []
        for idx1 in range(len(poly1)):
            for idx2 in range(idx1+1, len(poly1)):
                rotCenter = tr.FindRotationCenter([poly1[idx1], poly1[idx2]],
                                                  [poly2[idx1], poly2[idx2]])
                rotCenters.append(rotCenter)
                print(rotCenter)
                rcSum = list(np.array(rcSum) + np.array(rotCenter))

        rotCenterAvg = list(np.array(rcSum) / n_points1)
        rcShift = [ -int(rc) for rc in rotCenterAvg ]
        rcShift.reverse()
        img1 = imsup.copy_am_ph_image(self.display.image.prev)
        img2 = imsup.copy_am_ph_image(self.display.image)

        bufSz = max([abs(x) for x in rcShift])
        dirs = 'tblr'
        img1Pad = imsup.PadImage(img1, bufSz, 0.0, dirs)
        img2Pad = imsup.PadImage(img2, bufSz, 0.0, dirs)

        img1Rc = imsup.shift_am_ph_image(img1Pad, rcShift)
        img2Rc = imsup.shift_am_ph_image(img2Pad, rcShift)

        img1Rc = imsup.create_imgexp_from_img(img1Rc)
        img2Rc = imsup.create_imgexp_from_img(img2Rc)

        rotAngles = []
        for idx, p1, p2 in zip(range(n_points1), poly1, poly2):
            p1New = CalcNewCoords(p1, rotCenterAvg)
            p2New = CalcNewCoords(p2, rotCenterAvg)
            poly1[idx] = p1New
            poly2[idx] = p2New
            rotAngles.append(CalcRotAngle(p1New, p2New))

        rotAngleAvg = np.average(rotAngles)

        print('---- Rotation ----')
        print([ 'phi{0} = {1:.0f} deg\n'.format(idx + 1, angle) for idx, angle in zip(range(len(rotAngles)), rotAngles) ])
        print('------------------')
        print('Average rotation = {0:.2f} deg'.format(rotAngleAvg))

        self.shift = rcShift
        self.rot_angle = rotAngleAvg

        img2Rot = tr.RotateImageSki(img2Rc, rotAngleAvg)
        padSz = (img2Rot.width - img1Rc.width) // 2
        img1RcPad = imsup.PadImage(img1Rc, padSz, 0.0, 'tblr')

        img1RcPad.UpdateBuffer()
        img2Rot.UpdateBuffer()

        mag_factor = curr_img.width / img1RcPad.width
        img1_mag = tr.RescaleImageSki(img1RcPad, mag_factor)
        img2_mag = tr.RescaleImageSki(img2Rot, mag_factor)

        self.insert_img_after_curr(img1_mag)
        self.insert_img_after_curr(img2_mag)

        print('Rotation complete!')

    def auto_shift_image(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points1 = self.display.pointSets[curr_idx - 1]
        points2 = self.display.pointSets[curr_idx]
        n_points1 = len(points1)
        n_points2 = len(points2)

        if n_points1 != n_points2:
            print('Mark the same number of points on both images!')
            return

        set1 = [CalcRealCoords(img_width, pt1) for pt1 in points1]
        set2 = [CalcRealCoords(img_width, pt2) for pt2 in points2]

        shift_sum = np.zeros(2, dtype=np.int32)
        for pt1, pt2 in zip(set1, set2):
            shift = np.array(pt1) - np.array(pt2)
            shift_sum += shift

        shift_avg = list(shift_sum // n_points1)
        shift_avg.reverse()     # !!!
        self.shift = shift_avg

        shifted_img2 = imsup.shift_am_ph_image(curr_img, shift_avg)

        shifted_img2.name = curr_img.name + '_sh'
        self.insert_img_after_curr(shifted_img2)

    def reshift(self):
        curr_img = self.display.image
        shift = self.shift
        shifted_img = imsup.shift_am_ph_image(curr_img, shift)
        shifted_img.name = curr_img.name + '_sh'
        self.insert_img_after_curr(shifted_img)

    def rerotate(self):
        curr_img = self.display.image
        rot_angle = self.rot_angle
        rotated_img = tr.RotateImageSki(curr_img, rot_angle)
        rotated_img.name = curr_img.name + '_rot'
        self.insert_img_after_curr(rotated_img)

    def get_scale_ratio_from_images(self):
        if self.display.image.prev is None:
            print('There is no previous image!')
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
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points2 = self.display.pointSets[curr_idx]
        n_points2 = len(points2)

        if curr_img.prev is None or n_points2 == 0:
            print('Using manual magnification...')
            self.scale_factor = float(self.scale_factor_input.text())
            self.rescale_image()
            return

        ref_img = curr_img.prev
        points1 = self.display.pointSets[curr_idx - 1]
        n_points1 = len(points1)

        if n_points1 != n_points2:
            print('Mark the same number of points on both images!')
            return

        poly1 = [CalcRealCoords(img_width, pt1) for pt1 in points1]
        poly2 = [CalcRealCoords(img_width, pt2) for pt2 in points2]

        poly1_dists = []
        poly2_dists = []
        for i in range(len(poly1)):
            for j in range(i + 1, len(poly1)):
                poly1_dists.append(CalcDistance(poly1[i], poly1[j]))
                poly2_dists.append(CalcDistance(poly2[i], poly2[j]))

        scfs = [dist1 / dist2 for dist1, dist2 in zip(poly1_dists, poly2_dists)]
        scf_avg = np.average(scfs)
        self.scale_factor = scf_avg

        print('---- Scale factor ----')
        print(['scf{0} = {1:.2f}x\n'.format(idx + 1, mag) for idx, mag in zip(range(len(scfs)), scfs)])
        print('Average scale factor = {0:.2f}x'.format(scf_avg))
        print('------------------')

        magnified_img = tr.RescaleImageSki(curr_img, scf_avg)

        pad_sz = (magnified_img.width - curr_img.width) // 2
        if pad_sz > 0:
            padded_img1 = imsup.pad_img_from_ref(ref_img, magnified_img.width, 0.0, 'tblr')
            padded_img2 = imsup.copy_am_ph_image(magnified_img)
            resc_factor = ref_img.width / padded_img1.width
            resc_img1 = tr.RescaleImageSki(padded_img1, resc_factor)
            resc_img2 = tr.RescaleImageSki(padded_img2, resc_factor)
            resc_img2.prev, resc_img2.next = None, None
        else:
            resc_img1 = imsup.copy_am_ph_image(ref_img)
            resc_img1.prev, resc_img1.next = None, None
            resc_img2 = imsup.pad_img_from_ref(magnified_img, ref_img.width, 0.0, 'tblr')

        resc_img1.name = ref_img.name + '_resc1'
        resc_img2.name = curr_img.name + '_resc2'
        self.insert_img_after_curr(resc_img1)
        self.insert_img_after_curr(resc_img2)

        print('Image(s) rescaled!')

    def rescale_image(self):
        curr_img = self.display.image
        mag_img = tr.RescaleImageSki(curr_img, self.scale_factor)
        pad_sz = (mag_img.width - curr_img.width) // 2

        if pad_sz > 0:
            pad_img = imsup.pad_img_from_ref(curr_img, mag_img.width, 0.0, 'tblr')
            resc_factor = curr_img.width / pad_img.width
            resc_img = tr.RescaleImageSki(pad_img, resc_factor)
        else:
            resc_img = imsup.pad_img_from_ref(mag_img, curr_img.width, 0.0, 'tblr')

        resc_img.name = curr_img.name + '_resc'
        self.insert_img_after_curr(resc_img)
        print('Image rescaled!')

    def warp_image(self, more_accurate=False):
        curr_img = self.display.image
        curr_idx = self.display.image.numInSeries - 1
        real_points1 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx-1])
        real_points2 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx])
        user_points1 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points1)
        user_points2 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points2)

        self.warp_points = [ user_points1, user_points2 ]

        if more_accurate:
            n_div = const.n_div_for_warp
            frag_dim_size = curr_img.width // n_div

            # points #1
            grid_points1 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points1 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points1 ]

            for pt1 in user_points1:
                closest_node = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt1 ]
                grid_points1 = [ pt1 if grid_node == closest_node else grid_node for grid_node in grid_points1 ]

            src = np.array(grid_points1)

            # points #2
            grid_points2 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points2 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points2 ]
            for pt2 in user_points2:
                closestNode = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt2 ]
                grid_points2 = [ pt2 if gridNode == closestNode else gridNode for gridNode in grid_points2 ]

            dst = np.array(grid_points2)

        else:
            src = np.array(user_points1)
            dst = np.array(user_points2)

        img_warp = tr.WarpImage(curr_img, src, dst)

        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img_warp)
        tmp_img_list.UpdateLinks()
        self.display.pointSets.insert(curr_num, [])
        self.go_to_next_image()

    def rewarp(self):
        curr_img = self.display.image
        user_pts1 = self.warp_points[0]
        user_pts2 = self.warp_points[1]

        src = np.array(user_pts1)
        dst = np.array(user_pts2)

        warped_img = tr.WarpImage(curr_img, src, dst)
        self.insert_img_after_curr(warped_img)

    def insert_img_after_curr(self, img):
        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img)
        self.display.pointSets.insert(curr_num, [])
        tmp_img_list.UpdateLinks()
        self.go_to_next_image()
        self.preview_scroll.update_scroll_list(self.display.image)

    def rec_holo_no_ref_1(self):
        holo_img = self.display.image
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        holo_fft.name = 'fft_of_{0}'.format(holo_img.name)
        # self.display.image = rescale_image_buffer_to_window(holo_img, const.disp_dim)
        # holo_fft = rescale_image_buffer_to_window(holo_fft, const.disp_dim)
        self.insert_img_after_curr(holo_fft)
        self.log_scale_checkbox.setChecked(True)

    def rec_holo_no_ref_2(self):
        # general convention is (y, x), i.e. (r, c)
        holo_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[holo_fft.numInSeries - 1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(holo_fft.width, dpts)
        rpt1 = rpts[:2] # x, y
        rpt2 = rpts[2:] # x, y
        rpt1.reverse()  # r, c
        rpt2.reverse()  # r, c

        sband = np.copy(holo_fft.amPh.am[rpt1[0]:rpt2[0], rpt1[1]:rpt2[1]])
        apply_subpx_shift = self.subpixel_shift_checkbox.isChecked()
        sband_xy = holo.find_sideband_center(sband, orig=rpt1, subpx=apply_subpx_shift)

        # ---
        px_dim = holo_fft.px_dim
        img_dim = holo_fft.width
        mid_x = img_dim // 2
        dx_dim = 1 / (px_dim * img_dim)
        sbx, sby = sband_xy[0] - mid_x, sband_xy[1] - mid_x
        sb_xy_comp = np.complex(sbx * dx_dim, sby * dx_dim)
        R = np.abs(sb_xy_comp)
        ang = imsup.Degrees(np.angle(sb_xy_comp))
        print('R = {0:.3f} um-1\nAng = {1:.2f} deg'.format(R * 1e-6, ang))
        # ---

        ap_radius = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        mid = holo_fft.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        sband_img_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_rad=ap_radius, N_hann=hann_window)
        self.log_scale_checkbox.setChecked(True)
        self.insert_img_after_curr(sband_img_ap)

    def rec_holo_with_ref_2(self):
        ref_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[ref_fft.numInSeries - 1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(ref_fft.width, dpts)
        rpt1 = rpts[:2] # x, y
        rpt2 = rpts[2:] # x, y
        rpt1.reverse()  # r, c
        rpt2.reverse()  # r, c

        sband = np.copy(ref_fft.amPh.am[rpt1[0]:rpt2[0], rpt1[1]:rpt2[1]])
        apply_subpx_shift = self.subpixel_shift_checkbox.isChecked()
        sband_xy = holo.find_sideband_center(sband, orig=rpt1, subpx=apply_subpx_shift)

        ap_radius = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        mid = ref_fft.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        ref_sband_ap = holo.rec_holo_no_ref_2(ref_fft, shift, ap_rad=ap_radius, N_hann=hann_window)

        holo_img = self.display.image.next
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        holo_sband_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_rad=ap_radius, N_hann=hann_window)

        self.log_scale_checkbox.setChecked(True)
        self.insert_img_after_curr(ref_sband_ap)
        self.insert_img_after_curr(holo_sband_ap)

    def rec_holo_no_ref_3(self):
        sband_img = self.display.image
        rec_holo = holo.rec_holo_no_ref_3(sband_img)
        rec_holo.ReIm2AmPh()
        self.log_scale_checkbox.setChecked(False)
        self.insert_img_after_curr(rec_holo)

    # def rec_holo_no_ref(self):
    #     holo1 = self.display.image.prev
    #     holo2 = self.display.image
    #
    #     rec_holo2 = holo.rec_holo_no_ref(holo2)
    #
    #     curr_num = self.display.image.numInSeries
    #     tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
    #
    #     if holo1 is not None:
    #         rec_holo1 = holo.rec_holo_no_ref(holo1)
    #         tmp_img_list.insert(1, rec_holo1)
    #         tmp_img_list.insert(2, rec_holo2)
    #         self.display.pointSets.insert(curr_num, [])
    #         self.display.pointSets.insert(curr_num+1, [])
    #     else:
    #         tmp_img_list.insert(1, rec_holo2)
    #         self.display.pointSets.insert(curr_num, [])
    #
    #     tmp_img_list.UpdateLinks()
    #     self.go_to_next_image()

    def rec_holo_with_ref(self):
        pass

    def calc_phs_sum(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_sum = holo.calc_phase_sum(rec_holo1, rec_holo2)
        phs_sum = rescale_image_buffer_to_window(phs_sum, const.disp_dim)
        self.insert_img_after_curr(phs_sum)

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)
        phs_diff = rescale_image_buffer_to_window(phs_diff, const.disp_dim)
        self.insert_img_after_curr(phs_diff)

    def amplify_phase(self):
        curr_img = self.display.image
        curr_name = self.name_input.text()
        amp_factor = float(self.amp_factor_input.text())

        phs_amplified = imsup.copy_am_ph_image(curr_img)
        phs_amplified.amPh.ph *= amp_factor
        phs_amplified.update_cos_phase()
        phs_amplified.name = '{0}_x{1:.0f}'.format(curr_name, amp_factor)
        phs_amplified = rescale_image_buffer_to_window(phs_amplified, const.disp_dim)
        self.insert_img_after_curr(phs_amplified)
        self.cos_phs_radio_button.setChecked(True)

    def add_radians(self):
        curr_img = self.display.image
        curr_name = self.name_input.text()
        radians = float(self.radians2add_input.text())

        new_phs_img = imsup.copy_am_ph_image(curr_img)
        new_phs_img.amPh.ph += radians
        new_phs_img.update_cos_phase()
        new_phs_img.name = '{0}_+{1:.2f}rad'.format(curr_name, radians)
        new_phs_img = rescale_image_buffer_to_window(new_phs_img, const.disp_dim)
        self.insert_img_after_curr(new_phs_img)
        self.cos_phs_radio_button.setChecked(True)
        print('Added {0:.2f} rad to "{1}"'.format(radians, curr_name))

    # def remove_phase_gradient(self):
    #     curr_img = self.display.image
    #     curr_idx = curr_img.numInSeries - 1
    #     p1, p2, p3 = self.display.pointSets[curr_idx][:3]
    #     p1.append(curr_img.amPh.ph[p1[1], p1[0]])
    #     p2.append(curr_img.amPh.ph[p2[1], p2[0]])
    #     p3.append(curr_img.amPh.ph[p3[1], p3[0]])
    #     grad_plane = tr.Plane(0, 0, 0)
    #     grad_plane.getFromThreePoints(p1, p2, p3)
    #     # print(grad_plane.a, grad_plane.b, grad_plane.c)
    #     grad_arr = grad_plane.fillPlane(curr_img.height, curr_img.width)
    #     grad_img = imsup.ImageExp(curr_img.height, curr_img.width)
    #     grad_img.amPh.ph = np.copy(grad_arr)
    #     # print(grad_arr[p1[1], p1[0]])
    #     # print(grad_arr[p2[1], p2[0]])
    #     # print(grad_arr[p3[1], p3[0]])
    #     # print(p1[2], p2[2], p3[2])
    #     self.insert_img_after_curr(grad_img)

    def remove_phase_tilt(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        h, w = curr_img.amPh.am.shape
        phs = np.copy(curr_img.amPh.ph)

        # default points
        xy1 = [0, h//2]
        xy2 = [w-1, h//2]
        xy3 = [w//2, 0]
        xy4 = [w//2, h-1]

        n_usr_pts = len(self.display.pointSets[curr_idx])

        if n_usr_pts == 2:
            print('Removing local phase tilt...')
            dpt1, dpt2 = self.display.pointSets[curr_idx][:2]
            rpt1 = CalcRealTLCoords(w, dpt1)
            rpt2 = CalcRealTLCoords(w, dpt2)
            mid_x = (rpt1[0]+rpt2[0]) // 2
            mid_y = (rpt1[1]+rpt2[1]) // 2
            xy1 = [rpt1[0], mid_y]
            xy2 = [rpt2[0], mid_y]
            xy3 = [mid_x, rpt1[1]]
            xy4 = [mid_x, rpt2[1]]
        elif n_usr_pts == 4:
            print('Removing global phase tilt...')
            dpts = self.display.pointSets[curr_idx][:4]
            # dpts_f = [c for dpt in dpts for c in dpt]     # unpacking list of lists
            rpts = [CalcRealTLCoords(w, dpt) for dpt in dpts]
            xy1[0] = rpts[0][0]
            xy2[0] = rpts[1][0]
            xy3[1] = rpts[2][1]
            xy4[1] = rpts[3][1]
        else:
            print('Using default configuration... [To change it mark 2 points (local phase tilt) or 4 points (global phase tilt) and repeat procedure]')

        n_neigh = 10
        px1 = [xy1[0], tr.calc_avg_neigh(phs, x=xy1[0], y=xy1[1], nn=n_neigh)]
        px2 = [xy2[0], tr.calc_avg_neigh(phs, x=xy2[0], y=xy2[1], nn=n_neigh)]
        py1 = [xy3[1], tr.calc_avg_neigh(phs, x=xy3[0], y=xy3[1], nn=n_neigh)]
        py2 = [xy4[1], tr.calc_avg_neigh(phs, x=xy4[0], y=xy4[1], nn=n_neigh)]

        x_line = tr.Line(0, 0)
        y_line = tr.Line(0, 0)
        x_line.getFromPoints(px1, px2)
        y_line.getFromPoints(py1, py2)

        X = np.arange(0, w, dtype=np.float32)
        Y = np.arange(0, h, dtype=np.float32)
        X, Y = np.meshgrid(X, Y)
        phs_grad_x = x_line.a * X + x_line.b
        phs_grad_y = y_line.a * Y + y_line.b
        phs_grad_xy = phs_grad_x + phs_grad_y

        phs_grad_img = imsup.ImageExp(curr_img.height, curr_img.width)
        phs_grad_img.LoadPhsData(phs_grad_xy)
        phs_grad_img.name = '{0}_tilt'.format(curr_img.name)

        new_phs_img = imsup.copy_am_ph_image(curr_img)
        new_phs_img.amPh.ph -= phs_grad_xy
        new_phs_img.name = '{0}_minus_tilt'.format(curr_img.name)

        self.insert_img_after_curr(phs_grad_img)
        self.insert_img_after_curr(new_phs_img)

    def get_sideband_from_xy(self):
        curr_img = self.display.image

        # temporary test for subpixel shifting (#1)
        # test_img = imsup.ImageExp(768, 768)
        # test_list = []
        # val = 0
        # for i in range(768):
        #     if not i % 16:
        #         val = not val
        #     test_list.append([float(val)] * 768)
        # test_arr = np.array(test_list, dtype=np.float32)
        # test_img.amPh.am = np.copy(test_arr)
        # test_img.amPh.ph = np.copy(test_arr)
        # subpx_shx = float(self.sideband_x_input.text())
        # subpx_shy = float(self.sideband_y_input.text())
        # if subpx_shx == 0 and subpx_shy == 0:
        #     self.insert_img_after_curr(test_img)
        # else:
        #     subpx_shifted_img = holo.subpixel_shift(test_img, [subpx_shy, subpx_shx])
        #     self.insert_img_after_curr(subpx_shifted_img)

        # temporary test for subpixel shifting (#2)
        # subpx_shx = float(self.sideband_x_input.text())
        # subpx_shy = float(self.sideband_y_input.text())
        # subpx_shifted_img = holo.subpixel_shift(curr_img, [subpx_shy, subpx_shx])
        # self.insert_img_after_curr(subpx_shifted_img)

        sbx = float(self.sideband_x_input.text())
        sby = float(self.sideband_y_input.text())
        sband_xy = [sby, sbx]

        mid = curr_img.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        ap_radius= int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        sband_img_ap = holo.rec_holo_no_ref_2(curr_img, shift, ap_rad=ap_radius, N_hann=hann_window)
        self.log_scale_checkbox.setChecked(True)
        self.insert_img_after_curr(sband_img_ap)

    # temporary method?
    def do_all(self):
        ref_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[ref_fft.numInSeries - 1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(ref_fft.width, dpts)
        rpt1 = rpts[:2] # x, y
        rpt2 = rpts[2:] # x, y
        rpt1.reverse()  # r, c
        rpt2.reverse()  # r, c

        sband = np.copy(ref_fft.amPh.am[rpt1[0]:rpt2[0], rpt1[1]:rpt2[1]])
        apply_subpx_shift = self.subpixel_shift_checkbox.isChecked()
        sband_xy = holo.find_sideband_center(sband, orig=rpt1, subpx=apply_subpx_shift)

        ap_radius = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        mid = ref_fft.width // 2
        shift = [mid - sband_xy[0], mid - sband_xy[1]]

        ref_sband_ap = holo.rec_holo_no_ref_2(ref_fft, shift, ap_rad=ap_radius, N_hann=hann_window)

        holo_img = ref_fft.next
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        holo_sband_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_rad=ap_radius, N_hann=hann_window)

        rec_ref = holo.rec_holo_no_ref_3(ref_sband_ap)
        rec_holo = holo.rec_holo_no_ref_3(holo_sband_ap)
        rec_ref.ReIm2AmPh()
        rec_holo.ReIm2AmPh()

        # unwrapping
        new_ref_phs = tr.unwrap_phase(rec_ref.amPh.ph)
        new_holo_phs = tr.unwrap_phase(rec_holo.amPh.ph)
        rec_ref.amPh.ph = np.copy(new_ref_phs)
        rec_holo.amPh.ph = np.copy(new_holo_phs)

        rec_holo_corr = holo.calc_phase_diff(rec_ref, rec_holo)
        rec_holo_corr = rescale_image_buffer_to_window(rec_holo_corr, const.disp_dim)
        rec_holo_corr.name = 'ph_from_{0}'.format(holo_img.name)
        self.insert_img_after_curr(rec_holo_corr)
        self.log_scale_checkbox.setChecked(False)

    def swap_left(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            return
        curr_idx = curr_img.numInSeries - 1

        first_img = imsup.GetFirstImage(curr_img)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        imgs[curr_idx-1], imgs[curr_idx] = imgs[curr_idx], imgs[curr_idx-1]

        imgs[0].prev = None
        imgs[len(imgs)-1].next = None
        imgs[curr_idx-1].numInSeries = imgs[curr_idx].numInSeries
        imgs.UpdateLinks()

        ps = self.display.pointSets
        if len(ps[curr_idx-1]) > 0:
            ps[curr_idx-1], ps[curr_idx] = ps[curr_idx], ps[curr_idx-1]
        self.go_to_next_image()

    def swap_right(self):
        curr_img = self.display.image
        if curr_img.next is None:
            return
        curr_idx = curr_img.numInSeries - 1

        first_img = imsup.GetFirstImage(curr_img)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        imgs[curr_idx], imgs[curr_idx+1] = imgs[curr_idx+1], imgs[curr_idx]

        imgs[0].prev = None
        imgs[len(imgs)-1].next = None
        imgs[curr_idx].numInSeries = imgs[curr_idx+1].numInSeries
        imgs.UpdateLinks()

        ps = self.display.pointSets
        if len(ps[curr_idx]) > 0:
            ps[curr_idx], ps[curr_idx+1] = ps[curr_idx+1], ps[curr_idx]
        self.go_to_prev_image()

    def plot_profile(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        px_sz = curr_img.px_dim
        print(px_sz)
        points = self.display.pointSets[curr_idx][:2]
        points = np.array([ CalcRealCoords(curr_img.width, pt) for pt in points ])

        # find rotation center (center of the line)
        rot_center = np.average(points, 0).astype(np.int32)
        print('rotCenter = {0}'.format(rot_center))

        # find direction (angle) of the line
        dir_info = FindDirectionAngles(points[0], points[1])
        dir_angle = imsup.Degrees(dir_info[0])
        proj_dir = dir_info[2]
        print('dir angle = {0:.2f} deg'.format(dir_angle))

        # shift image by -center
        shift_to_rot_center = list(-rot_center)
        shift_to_rot_center.reverse()
        img_shifted = imsup.shift_am_ph_image(curr_img, shift_to_rot_center)

        # rotate image by angle
        img_rot = tr.RotateImageSki(img_shifted, dir_angle)

        # crop fragment (height = distance between two points)
        pt_diffs = points[0] - points[1]
        frag_dim1 = int(np.sqrt(pt_diffs[0] ** 2 + pt_diffs[1] ** 2))
        frag_dim2 = int(self.int_width_input.text())

        if proj_dir == 0:
            frag_width, frag_height = frag_dim1, frag_dim2
        else:
            frag_width, frag_height = frag_dim2, frag_dim1

        frag_coords = imsup.DetermineCropCoordsForNewDims(img_rot.width, img_rot.height, frag_width, frag_height)
        print('Frag dims = {0}, {1}'.format(frag_width, frag_height))
        print('Frag coords = {0}'.format(frag_coords))
        img_cropped = imsup.crop_am_ph_roi(img_rot, frag_coords)

        # calculate projection of intensity
        if self.amp_radio_button.isChecked():
            int_matrix = np.copy(img_cropped.amPh.am)
        elif self.phs_radio_button.isChecked():
            # ph_min = np.min(img_cropped.amPh.ph)
            # ph_fix = -ph_min if ph_min < 0 else 0
            # img_cropped.amPh.ph += ph_fix
            int_matrix = np.copy(img_cropped.amPh.ph)
        else:
            # cos_ph_min = np.min(img_cropped.cos_phase)
            # cos_ph_fix = -cos_ph_min if cos_ph_min < 0 else 0
            # img_cropped.cos_phase += cos_ph_fix
            int_matrix = np.copy(img_cropped.cos_phase)

        int_profile = np.sum(int_matrix, proj_dir)  # 0 - horizontal projection, 1 - vertical projection
        dists = np.arange(0, int_profile.shape[0], 1) * px_sz
        dists *= 1e9

        self.plot_widget.plot(dists, int_profile, 'Distance [nm]', 'Intensity [a.u.]')

    def calc_phase_gradient(self):
        curr_img = self.display.image
        dx_img = imsup.copy_am_ph_image(curr_img)
        dy_img = imsup.copy_am_ph_image(curr_img)
        grad_img = imsup.copy_am_ph_image(curr_img)
        print('Calculating gradient for sample distance = {0:.2f} nm'.format(curr_img.px_dim * 1e9))
        dx, dy = np.gradient(curr_img.amPh.ph, curr_img.px_dim)
        dr = np.sqrt(dx * dx + dy * dy)
        # dphi = np.arctan2(dy, dx)
        dx_img.amPh.ph = np.copy(dx)
        dy_img.amPh.ph = np.copy(dy)
        grad_img.amPh.ph = np.copy(dr)
        dx_img.name = 'gradX_of_{0}'.format(curr_img.name)
        dy_img.name = 'gradY_of_{0}'.format(curr_img.name)
        grad_img.name = 'gradM_of_{0}'.format(curr_img.name)
        self.insert_img_after_curr(dx_img)
        self.insert_img_after_curr(dy_img)
        self.insert_img_after_curr(grad_img)

    def calc_Bxy_maps(self):
        curr_img = self.display.image
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_coeff = const.dirac_const / sample_thickness

        dx, dy = np.gradient(curr_img.amPh.ph, curr_img.px_dim)
        # B_sign = np.sign(dx)
        # B_field = B_sign * np.sqrt(dx * dx + dy * dy) * B_coeff
        Bx = B_coeff * dx
        By = B_coeff * dy

        Bx_img = imsup.copy_am_ph_image(curr_img)
        Bx_img.amPh.am *= 0
        Bx_img.amPh.ph = np.copy(Bx)
        Bx_img.name = 'Bx_from_{0}'.format(curr_img.name)

        By_img = imsup.copy_am_ph_image(Bx_img)
        By_img.amPh.am *= 0
        By_img.amPh.ph = np.copy(By)
        By_img.name = 'By_from_{0}'.format(curr_img.name)

        self.insert_img_after_curr(Bx_img)
        self.insert_img_after_curr(By_img)

    def calc_B_polar_from_section(self):
        from numpy import linalg as la
        curr_img = self.display.image
        curr_phs = curr_img.amPh.ph
        curr_idx = curr_img.numInSeries - 1
        px_sz = curr_img.px_dim
        dpt1, dpt2 = self.display.pointSets[curr_idx][:2]
        pt1 = np.array(CalcRealTLCoords(curr_img.width, dpt1))
        pt2 = np.array(CalcRealTLCoords(curr_img.width, dpt2))

        d_dist = la.norm(pt1 - pt2) # * px_sz
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        pt1_is_orig = self.orig_in_pt1_radio_button.isChecked()

        if pt1_is_orig:
            orig_xy = pt1
            r1 = int(d_dist)
        else:
            orig_xy = np.array([int(np.mean((pt1[0], pt2[0]))), int(np.mean((pt1[1], pt2[1])))])
            r1 = int(d_dist // 2)

        n_r = int(self.num_of_r_iters_input.text())
        min_xy = [ x - n_r * r1 for x in orig_xy ]
        max_xy = [ x + n_r * r1 for x in orig_xy ]
        if min_xy[0] < 0 or min_xy[1] < 0 or max_xy[0] > curr_img.width or max_xy[1] > curr_img.height:
            print('One of the circular areas will go beyond edges of the image. Select new area or lower the number of radius iterations.')
            return

        r_values = [ (n + 1) * r1 for n in range(n_r) ]
        print('r = {0}'.format(r_values))

        ang0 = -np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        ang1 = ang0 - np.pi/2.0
        ang2 = ang0 + np.pi/2.0
        n_ang = 60
        ang_arr = np.linspace(ang1, ang2, n_ang, dtype=np.float32)
        angles = [ np.copy(ang_arr) for _ in range(n_r) ]

        B_coeff = const.dirac_const / (sample_thickness * d_dist * px_sz)
        B_values = [ [] for _ in range(n_r) ]
        x_arr_for_ls = np.linspace(0, d_dist * px_sz, 5, dtype=np.float32)

        for r, r_idx in zip(r_values, range(n_r)):
            nn_for_ls = 4 if r >= 16 else r // 4

            for ang, a_idx in zip(angles[r_idx], range(n_ang)):
                sin_cos = np.array([np.cos(ang), -np.sin(ang)])         # -sin(ang), because y increases from top to bottom of an image
                new_pt1 = pt1 if pt1_is_orig else np.array(orig_xy - r * sin_cos).astype(np.int32)
                new_pt2 = np.array(orig_xy + r * sin_cos).astype(np.int32)
                x1, y1 = new_pt1
                x2, y2 = new_pt2

                xx = np.linspace(x1, x2, 5, dtype=np.int32)
                yy = np.linspace(y1, y2, 5, dtype=np.int32)
                # ph_arr_for_ls = [ curr_phs[y, x] for y, x in zip(yy, xx) ]
                ph_arr_for_ls = [ tr.calc_avg_neigh(curr_phs, x, y, nn=nn_for_ls) for x, y in zip(xx, yy) ]
                aa, bb = tr.LinLeastSquares(x_arr_for_ls, ph_arr_for_ls)

                # d_phase = tr.calc_avg_neigh(curr_phs, x1, y1, nn=4) - tr.calc_avg_neigh(curr_phs, x2, y2, nn=4)
                d_phase = aa * (x_arr_for_ls[4] - x_arr_for_ls[0])
                B_val = B_coeff * d_phase
                if B_val < 0: angles[r_idx][a_idx] += np.pi
                B_values[r_idx].append(np.abs(B_val))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')

        B_min, B_max = 0.0, np.max(B_values) + 0.1
        # B_min, B_max = 0.0, const.temp_B_max_for_polar_plot
        for p_idx in range(n_r):
            ax.plot(angles[p_idx], np.array(B_values[p_idx]), '.-', lw=1.0, ms=3.5, label='r={0}px'.format(r_values[p_idx]))
        ax.plot(np.array([ang0, ang0 + np.pi]), np.array([B_max, B_max]), 'k--', lw=0.8)    # mark selected direction
        ax.plot(np.array([ang1, ang2]), np.array([B_max, B_max]), 'g--', lw=0.8)            # boundary between positive and negative values of B
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        # ax.plot(angles, np.zeros(n_ang), 'g--', lw=1)
        # for ang, r in zip(angles[:n_ang:4], B_values[:n_ang:4]):
        #     ax.annotate('', xytext=(0.0, r_min), xy=(ang, r), arrowprops=dict(facecolor='blue', arrowstyle='->'))
        ax.set_ylim(B_min, B_max)
        ax.grid(True)

        plt.margins(0, 0)
        plt.savefig('B_pol_{0}.png'.format(curr_img.name), dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.clf()
        # plt.cla()
        plt.close(fig)
        print('B_pol_{0}.png exported!'.format(curr_img.name))

    def calc_B_polar_from_area(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        px_sz = curr_img.px_dim
        if len(self.display.pointSets[curr_idx]) < 2:
            print('You have to mark two points!')
            return

        pt1, pt2 = self.display.pointSets[curr_idx][:2]
        pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
        disp_crop_coords = pt1 + pt2
        real_tl_coords = CalcRealTLCoords(curr_img.width, disp_crop_coords)
        real_sq_coords = imsup.MakeSquareCoords(real_tl_coords)
        frag = zoom_fragment(curr_img, real_sq_coords)

        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_coeff = const.dirac_const / sample_thickness
        angles = np.arange(0, 361, 10, dtype=np.float32)
        B_means = []

        for ang in angles:
            frag_rot = tr.RotateImageSki(frag, -ang)
            dx, dy = np.gradient(frag_rot.amPh.ph, px_sz)
            B_mat = B_coeff * dx
            n_el = np.count_nonzero(B_mat)
            B_mean = np.sum(B_mat) / n_el
            B_means.append(B_mean)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        rad_angles = imsup.Radians(angles)
        ax.plot(rad_angles, np.array(B_means))
        ax.set_rmin(-0.5)
        ax.set_rmax(0.5)
        ax.grid(True)

        plt.margins(0, 0)
        plt.savefig('B_pol_{0}.png'.format(curr_img.name), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.cla()
        plt.close(fig)
        print('B_pol_{0}.png exported!'.format(curr_img.name))

    # calculate B from section on image
    def calc_B_from_section(self):
        from numpy import linalg as la
        curr_img = self.display.image
        curr_phs = curr_img.amPh.ph
        curr_idx = curr_img.numInSeries - 1
        px_sz = curr_img.px_dim
        dpt1, dpt2 = self.display.pointSets[curr_idx][:2]
        pt1 = np.array(CalcRealTLCoords(curr_img.width, dpt1))
        pt2 = np.array(CalcRealTLCoords(curr_img.width, dpt2))

        d_dist = la.norm(pt1-pt2) * px_sz
        # d_phase = np.abs(curr_phs[pt1[1], pt1[0]] - curr_phs[pt2[1], pt2[0]])
        # ---
        ph1_avg = tr.calc_avg_neigh(curr_phs, pt1[0], pt1[1], nn=10)
        ph2_avg = tr.calc_avg_neigh(curr_phs, pt2[0], pt2[1], nn=10)
        d_phase = ph2_avg - ph1_avg         # consider sign of magnetic field
        # d_phase = np.abs(ph1_avg - ph2_avg)
        # ---
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_in_plane = (const.dirac_const / sample_thickness) * (d_phase / d_dist)
        print('{0:.1f} nm'.format(d_dist * 1e9))
        print('{0:.2f} rad'.format(d_phase))
        print('B = {0:.2f} T'.format(B_in_plane))

    # calculate B from profile in PlotWidget
    def calc_B_from_profile(self):
        pt1, pt2 = self.plot_widget.markedPointsData
        d_dist = np.abs(pt1[0] - pt2[0]) * 1e-9
        d_phase = np.abs(pt1[1] - pt2[1])
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_in_plane = (const.dirac_const / sample_thickness) * (d_phase / d_dist)
        print('{0:.1f} nm'.format(d_dist * 1e9))
        print('{0:.2f} rad'.format(d_phase))
        print('B = {0:.2f} T'.format(B_in_plane))

    def gen_phase_stats(self):
        curr_img = self.display.image
        curr_phs = curr_img.amPh.ph
        print('STATISTICS for phase of "{0}":'.format(curr_img.name))
        print('Min. = {0:.2f}\nMax. = {1:.2f}\nAvg. = {2:.2f}'.format(np.min(curr_phs), np.max(curr_phs), np.mean(curr_phs)))
        print('Med. = {0:.2f}\nStd. dev. = {1:.2f}\nVar. = {2:.2f}'.format(np.median(curr_phs), np.std(curr_phs), np.var(curr_phs)))

        if curr_img.prev is not None:
            max_shift = const.corr_arr_max_shift
            print('Calculating ({0}x{0}) correlation array between *curr* and *prev* phases...'.format(2 * max_shift))
            prev_img = curr_img.prev
            prev_phs = prev_img.amPh.ph
            corr_arr = tr.calc_corr_array(prev_phs, curr_phs, max_shift)
            ch, cw = corr_arr.shape
            corr_img = imsup.ImageExp(ch, cw)
            corr_img.LoadAmpData(corr_arr)
            corr_img.LoadPhsData(corr_arr)
            corr_img = rescale_image_buffer_to_window(corr_img, const.disp_dim)
            corr_img.name = 'corr_arr_{0}_vs_{1}'.format(curr_img.name, prev_img.name)
            self.insert_img_after_curr(corr_img)

            # single-value correlation coefficient
            # p1 = prev_phs - np.mean(prev_phs)
            # p2 = curr_phs - np.mean(curr_phs)
            # corr_coef = np.sum(p1 * p2) / np.sqrt(np.sum(p1 * p1) * np.sum(p2 * p2))
            # print('Corr. coef. between *curr* and *prev* phases = {0:.4f}'.format(corr_coef))

            # Pearson correlation matrix
            # corr_coef_arr = np.corrcoef(prev_phs, curr_phs)
            # corr_coef_img = imsup.ImageExp(curr_img.height, curr_img.width, px_dim_sz=curr_img.px_dim)
            # corr_coef_img.amPh.ph = np.copy(corr_coef_arr)
            # corr_coef_img = rescale_image_buffer_to_window(corr_coef_img, const.disp_dim)
            # corr_coef_img.name = 'corr_coef_{0}_vs_{1}'.format(curr_img.name, prev_img.name)
            # self.insert_img_after_curr(corr_coef_img)

    def calc_mean_inner_potential(self):
        curr_img = self.display.image
        curr_phs = curr_img.amPh.ph

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

        mean_inner_pot_img = imsup.copy_am_ph_image(curr_img)
        mean_inner_pot_img.amPh.am *= 0
        mean_inner_pot_img.amPh.ph = np.copy(mip)
        mean_inner_pot_img.name = 'MIP_from_{0}'.format(curr_img.name)
        self.insert_img_after_curr(mean_inner_pot_img)

    def filter_contours(self):
        curr_img = self.display.image
        conts = np.copy(curr_img.cos_phase)
        conts_scaled = imsup.ScaleImage(conts, 0, 1)
        threshold = float(self.threshold_input.text())
        conts_scaled[conts_scaled < threshold] = 0
        img_filtered = imsup.copy_am_ph_image(curr_img)
        img_filtered.amPh.ph = np.copy(conts_scaled)
        self.insert_img_after_curr(img_filtered)
        # find_contours(self.display.image)

    # def draw_image_with_arrows(self):
    #     import GradientArrows as grad_arr
    #     curr_img = self.display.image
    #     arr = np.copy(curr_img.amPh.ph)
    #     print('ble1')
    #     grad_arr.draw_image_with_gradient_arrows(arr, 20)

    # def plot_profile(self):
    #     curr_img = self.display.image
    #     curr_idx = curr_img.numInSeries - 1
    #     px_sz = curr_img.px_dim
    #     p1, p2 = self.display.pointSets[curr_idx][:2]
    #     p1 = CalcRealCoords(curr_img.width, p1)
    #     p2 = CalcRealCoords(curr_img.width, p2)
    #
    #     x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
    #     y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
    #     x_dist = x2 - x1
    #     y_dist = y2 - y1
    #
    #     if x_dist > y_dist:
    #         x_range = list(range(x1, x2))
    #         a_coeff = (p2[1] - p1[1]) / (p2[0] - p1[0])
    #         b_coeff = p1[1] - a_coeff * p1[0]
    #         y_range = [ int(a_coeff * x + b_coeff) for x in x_range ]
    #     else:
    #         y_range = list(range(y1, y2))
    #         a_coeff = (p2[0] - p1[0]) / (p2[1] - p1[1])
    #         b_coeff = p1[0] - a_coeff * p1[1]
    #         x_range = [ int(a_coeff * y + b_coeff) for y in y_range ]
    #
    #     print(len(x_range), len(y_range))
    #     profile = curr_img.amPh.am[x_range, y_range]
    #     dists = np.arange(0, profile.shape[0], 1) * px_sz
    #     dists *= 1e9
    #     self.plot_widget.plot(dists, profile, 'Distance [nm]', 'Intensity [a.u.]')

# --------------------------------------------------------

def open_dm3_file(file_path, img_type='amp'):
    img_data, px_dims = dm3.ReadDm3File(file_path)
    imsup.Image.px_dim_default = px_dims[0]
    new_img = imsup.ImageExp(img_data.shape[0], img_data.shape[1], imsup.Image.cmp['CAP'], px_dim_sz=px_dims[0])

    if img_type == 'amp':
        # amp_data = np.sqrt(np.abs(img_data))
        amp_data = np.copy(img_data)
        new_img.LoadAmpData(amp_data.astype(np.float32))
    else:
        new_img.LoadPhsData(img_data.astype(np.float32))

    return new_img

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(img_path):
    img_list = imsup.ImageList()
    img_num_match = re.search('([0-9]+).dm3', img_path)
    img_num_text = img_num_match.group(1)
    img_num = int(img_num_text)

    img_idx = 0
    is_there_info = False
    imgs_info = None

    if img_idx == 0:
        import pandas as pd
        first_img_name_match = re.search('(.+)/(.+).dm3$', img_path)
        dir_path = first_img_name_match.group(1)
        info_file_path = '{0}/info.txt'.format(dir_path)
        if path.isfile(info_file_path):
            is_there_info = True
            imgs_info = pd.read_csv(info_file_path, sep='\t', header=None)
            imgs_info = imgs_info.values

    while path.isfile(img_path):
        print('Reading file "' + img_path + '"')
        img_name_match = re.search('(.+)/(.+).dm3$', img_path)
        img_name_text = img_name_match.group(2)
        img_type = imgs_info[img_idx, 2]

        img = open_dm3_file(img_path, img_type)
        img.numInSeries = img_num
        img.name = img_name_text if not is_there_info else imgs_info[img_idx, 1]
        img = rescale_image_buffer_to_window(img, const.disp_dim)

        # ---
        # imsup.RemovePixelArtifacts(img, const.min_px_threshold, const.max_px_threshold)
        # imsup.RemovePixelArtifacts(img, 0.7, 1.3)
        # img.UpdateBuffer()
        # ---
        img_list.append(img)

        img_idx += 1
        img_num += 1
        img_num_text_new = img_num_text.replace(str(img_num-1), str(img_num))
        if img_num == 10:
            img_num_text_new = img_num_text_new[1:]
        img_path = rreplace(img_path, img_num_text, img_num_text_new, 1)
        img_num_text = img_num_text_new

    img_list.UpdateLinks()
    return img_list[0]

# --------------------------------------------------------

def rescale_image_buffer_to_window(img, win_dim):
    zoom_factor = win_dim / img.width
    img_to_disp = tr.RescaleImageSki(img, zoom_factor)
    img.buffer = imsup.ComplexAmPhMatrix(img_to_disp.height, img_to_disp.width)
    img.buffer.am = np.copy(img_to_disp.amPh.am)
    img.buffer.ph = np.copy(img_to_disp.amPh.ph)
    return img

# --------------------------------------------------------

def cross_corr_images(img_list):
    img_align_list = imsup.ImageList()
    img_list[0].shift = [0, 0]
    for img in img_list[1:]:
        mcf = imsup.CalcCrossCorrFun(img.prev, img)
        new_shift = imsup.GetShift(mcf)
        img.shift = [ sp + sn for sp, sn in zip(img.prev.shift, new_shift) ]
        # img.shift = list(np.array(img.shift) + np.array(new_shift))
        print('"{0}" was shifted by {1} px'.format(img.name, img.shift))
        img_shifted = imsup.shift_am_ph_image(img, img.shift)
        img_align_list.append(img_shifted)
    return img_align_list

# --------------------------------------------------------

def zoom_fragment(img, coords):
    crop_img = imsup.crop_am_ph_roi(img, coords)
    crop_img = imsup.create_imgexp_from_img(crop_img)
    # crop_img.MoveToCPU()

    crop_img.defocus = img.defocus
    crop_img = rescale_image_buffer_to_window(crop_img, const.disp_dim)
    return crop_img

# --------------------------------------------------------

def modify_image(img, mod=list([0, 0]), is_shift=True):
    if is_shift:
        mod_img = imsup.shift_am_ph_image(img, mod)
    else:
        mod_img = tr.RotateImageSki(img, mod[0])

    return mod_img

# --------------------------------------------------------

def norm_phase_to_pt(phase, pt):
    x, y = pt
    phase_norm = phase - phase[y, x]
    return phase_norm

# --------------------------------------------------------

def norm_phase_to_area(phase, pt1, pt2):
    # pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
    x1, y1 = pt1
    x2, y2 = pt2
    phs_avg = np.average(phase[y1:y2, x1:x2])
    phase_norm = phase - phs_avg
    return phase_norm

# --------------------------------------------------------

def FindDirectionAngles(p1, p2):
    lpt = p1[:] if p1[0] < p2[0] else p2[:]     # left point
    rpt = p1[:] if p1[0] > p2[0] else p2[:]     # right point
    dx = np.abs(rpt[0] - lpt[0])
    dy = np.abs(rpt[1] - lpt[1])
    sign = 1 if rpt[1] < lpt[1] else -1
    projDir = 1         # projection on y axis
    if dx > dy:
        sign *= -1
        projDir = 0     # projection on x axis
    diff1 = dx if dx < dy else dy
    diff2 = dx if dx > dy else dy
    ang1 = np.arctan2(diff1, diff2)
    ang2 = np.pi / 2 - ang1
    ang1 *= sign
    ang2 *= (-sign)
    return ang1, ang2, projDir

# --------------------------------------------------------

def CalcTopLeftCoords(imgWidth, midCoords):
    topLeftCoords = [ mc + imgWidth // 2 for mc in midCoords ]
    return topLeftCoords

# --------------------------------------------------------

def CalcTopLeftCoordsForSetOfPoints(imgWidth, points):
    topLeftPoints = [ CalcTopLeftCoords(imgWidth, pt) for pt in points ]
    return topLeftPoints

# --------------------------------------------------------

def CalcRealTLCoords(imgWidth, dispCoords):
    dispWidth = const.disp_dim
    factor = imgWidth / dispWidth
    realCoords = [ int(dc * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealTLCoordsForSetOfPoints(imgWidth, points):
    realCoords = [ CalcRealTLCoords(imgWidth, pt) for pt in points ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoords(imgWidth, dispCoords):
    dispWidth = const.disp_dim
    factor = imgWidth / dispWidth
    realCoords = [ int((dc - dispWidth // 2) * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoordsForSetOfPoints(imgWidth, points):
    realPoints = [ CalcRealCoords(imgWidth, pt) for pt in points ]
    return realPoints

# --------------------------------------------------------

def CalcRealTLCoordsForPaddedImage(imgWidth, dispCoords):
    dispWidth = const.disp_dim
    padImgWidthReal = np.ceil(imgWidth / 512.0) * 512.0
    pad = (padImgWidthReal - imgWidth) / 2.0
    factor = padImgWidthReal / dispWidth
    # dispPad = pad / factor
    # realCoords = [ (dc - dispPad) * factor for dc in dispCoords ]
    realCoords = [ int(dc * factor - pad) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcDispCoords(dispWidth, imgWidth, realCoords):
    factor = dispWidth / imgWidth
    dispCoords = [ (rc * factor) + const.disp_dim // 2 for rc in realCoords ]
    return dispCoords

# --------------------------------------------------------

def CalcDistance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

# --------------------------------------------------------

def CalcInnerAngle(a, b, c):
    alpha = np.arccos(np.abs((a*a + b*b - c*c) / (2*a*b)))
    return imsup.Degrees(alpha)

# --------------------------------------------------------

def CalcOuterAngle(p1, p2):
    dist = CalcDistance(p1, p2)
    betha = np.arcsin(np.abs(p1[0] - p2[0]) / dist)
    return imsup.Degrees(betha)

# --------------------------------------------------------

def CalcNewCoords(p1, newCenter):
    p2 = [ px - cx for px, cx in zip(p1, newCenter) ]
    return p2

# --------------------------------------------------------

def CalcRotAngle(p1, p2):
    z1 = np.complex(p1[0], p1[1])
    z2 = np.complex(p2[0], p2[1])
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    # rotAngle = np.abs(imsup.Degrees(phi2 - phi1))
    rotAngle = imsup.Degrees(phi2 - phi1)
    if np.abs(rotAngle) > 180:
        rotAngle = -np.sign(rotAngle) * (360 - np.abs(rotAngle))
    return rotAngle

# --------------------------------------------------------

def convert_points_to_tl_br(p1, p2):
    tl = list(np.amin([p1, p2], axis=0))
    br = list(np.amax([p1, p2], axis=0))
    return tl, br

# --------------------------------------------------------

def det_Imin_Imax_from_contrast(dI, def_max=256.0):
    dImin = dI // 2 + 1
    dImax = dI - dImin
    Imin = def_max // 2 - dImin
    Imax = def_max // 2 + dImax
    return Imin, Imax

# --------------------------------------------------------

def switch_xy(xy):
    return [xy[1], xy[0]]

# --------------------------------------------------------

def rreplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def export_glob_sc_images(img_list, add_arrows=True, rot_by_90=False, arr_size=20, arr_dist=50, cbar_lab=''):
    global_limits = [1e5, 0]

    for img in img_list:
        limits = [np.min(img.amPh.ph), np.max(img.amPh.ph)]
        if limits[0] < global_limits[0]:
            global_limits[0] = limits[0]
        if limits[1] > global_limits[1]:
            global_limits[1] = limits[1]

    fig = plt.figure()
    for img, idx in zip(img_list, range(1, len(img_list)+1)):
        plt.imshow(img.amPh.ph, vmin=global_limits[0], vmax=global_limits[1], cmap=plt.cm.get_cmap('jet'))

        if idx == len(img_list):
            cbar = plt.colorbar(label=cbar_lab)
            cbar.set_label(cbar_lab)

        if add_arrows:
            width, height = img.amPh.ph.shape
            xv, yv = np.meshgrid(np.arange(0, width, float(arr_dist)), np.arange(0, height, float(arr_dist)))
            xv += arr_dist / 2.0
            yv += arr_dist / 2.0

            phd = img.amPh.ph[0:height:arr_dist, 0:width:arr_dist]
            yd, xd = np.gradient(phd)

            # arrows along magnetic contours
            if rot_by_90:
                xd_yd_comp = xd + 1j * yd
                xd_yd_comp_rot = xd_yd_comp * np.exp(1j * np.pi / 2.0)
                xd = xd_yd_comp_rot.real
                yd = xd_yd_comp_rot.imag

            vectorized_arrow_drawing = np.vectorize(func_to_vectorize)
            vectorized_arrow_drawing(xv, yv, xd, yd, arr_size)

        out_f = '{0}.png'.format(img.name)
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(out_f, dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.clf()
        plt.cla()
    plt.close(fig)

# --------------------------------------------------------

def RunHolographyWindow():
    app = QtWidgets.QApplication(sys.argv)
    holo_window = HolographyWindow()
    sys.exit(app.exec_())