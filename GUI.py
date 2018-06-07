import re
import sys
from os import path
from functools import partial
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
import Dm3Reader3 as dm3
import Constants as const
import ImageSupport as imsup
import CrossCorr as cc
import Transform as tr
import Holo as holo

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# --------------------------------------------------------

class LabelExt(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        self.image = image
        self.setImage()
        self.pointSets = [[]]
        self.show_lines = True
        self.show_labs = True
        # while image.next is not None:
        #    self.pointSets.append([])

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
        if self.show_lines:
            linePen.setWidth(2)
            qp.setPen(linePen)
            for pt1, pt2 in zip(self.pointSets[imgIdx], self.pointSets[imgIdx][1:] + self.pointSets[imgIdx][:1]):
                line = QtCore.QLine(pt1[0], pt1[1], pt2[0], pt2[1])
                qp.drawLine(line)
        qp.end()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.x(), pos.y()]
        self.pointSets[self.image.numInSeries - 1].append(currPos)
        self.repaint()

        if self.parent().show_labels_checkbox.isChecked():
            lab = QtWidgets.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries - 1])), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pos.x()+4, pos.y()+4)
            lab.show()

    def setImage(self, dispAmp=True, dispPhs=False, logScale=False, color=False):
        self.image.MoveToCPU()

        if dispAmp:
            self.image.buffer = np.copy(self.image.amPh.am)
            if logScale:
                buf = self.image.buffer
                buf[np.where(buf <= 0)] = 1e-5
                self.image.buffer = np.log(buf)
        elif dispPhs:
            self.image.buffer = np.copy(self.image.amPh.ph)
        else:
            if self.image.cos_phase is None:
                self.image.update_cos_phase()
            self.image.buffer = np.copy(self.image.cos_phase)

        q_image = QtGui.QImage(imsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
                               self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)

        if color:
            step = 6
            inc_range = np.arange(0, 256, step)
            dec_range = np.arange(255, -1, -step)
            bcm1 = [ QtGui.qRgb(0, i, 255) for i in inc_range ]
            gcm1 = [ QtGui.qRgb(0, 255, i) for i in dec_range ]
            gcm2 = [ QtGui.qRgb(i, 255, 0) for i in inc_range ]
            rcm1 = [ QtGui.qRgb(255, i, 0) for i in dec_range ]
            rcm2 = [ QtGui.qRgb(255, 0, i) for i in inc_range ]
            bcm2 = [ QtGui.qRgb(i, 0, 255) for i in dec_range ]
            cm = bcm1 + gcm1 + gcm2 + rcm1 + rcm2 + bcm2
            q_image.setColorTable(cm)

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)
        self.repaint()

    def changeImage(self, toNext=True, dispAmp=True, dispPhs=False, logScale=False, dispLabs=True, color=False):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is None:
            return

        newImage.ReIm2AmPh()
        self.image = newImage

        if len(self.pointSets) < self.image.numInSeries:
            self.pointSets.append([])
        self.setImage(dispAmp, dispPhs, logScale, color)

        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

        if dispLabs:
            self.show_labels()

    def hide_labels(self):
        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

    def show_labels(self):
        imgIdx = self.image.numInSeries - 1
        for pt, idx in zip(self.pointSets[imgIdx], range(1, len(self.pointSets[imgIdx]) + 1)):
            lab = QtWidgets.QLabel('{0}'.format(idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pt[0] + 4, pt[1] + 4)
            lab.show()

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

class TriangulateWidget(QtWidgets.QWidget):
    def __init__(self):
        super(TriangulateWidget, self).__init__()
        file_dialog = QtWidgets.QFileDialog()
        image_path = file_dialog.getOpenFileName()[0]
        if image_path == '':
            print('No images to read. Exiting...')
            exit()
        image = LoadImageSeriesFromFirstFile(image_path)
        self.display = LabelExt(self, image)
        self.plot_widget = PlotWidget()
        self.backup_image = None
        self.changes_made = []
        self.shift = [0, 0]
        self.rot_angle = 0
        self.mag_coeff = 1.0
        self.warp_points = []
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(250)

        prev_button = QtWidgets.QPushButton('Prev', self)
        next_button = QtWidgets.QPushButton('Next', self)

        prev_button.clicked.connect(self.go_to_prev_image)
        next_button.clicked.connect(self.go_to_next_image)

        lswap_button = QtWidgets.QPushButton('L-Swap', self)
        rswap_button = QtWidgets.QPushButton('R-Swap', self)

        lswap_button.clicked.connect(self.swap_left)
        rswap_button.clicked.connect(self.swap_right)

        flip_button = QtWidgets.QPushButton('Flip', self)

        name_it_button = QtWidgets.QPushButton('Name it!', self)
        self.name_input = QtWidgets.QLineEdit('ref', self)

        zoom_button = QtWidgets.QPushButton('Zoom N images', self)
        self.n_to_zoom_input = QtWidgets.QLineEdit('1', self)

        hbox_name = QtWidgets.QHBoxLayout()
        hbox_name.addWidget(name_it_button)
        hbox_name.addWidget(self.name_input)

        hbox_zoom = QtWidgets.QHBoxLayout()
        hbox_zoom.addWidget(zoom_button)
        hbox_zoom.addWidget(self.n_to_zoom_input)

        name_it_button.setFixedWidth(115)
        zoom_button.setFixedWidth(115)
        self.name_input.setFixedWidth(115)
        self.n_to_zoom_input.setFixedWidth(115)

        flip_button.clicked.connect(self.flip_image_h)
        name_it_button.clicked.connect(self.set_image_name)
        zoom_button.clicked.connect(self.zoom_n_fragments)

        export_button = QtWidgets.QPushButton('Export', self)
        export_all_button = QtWidgets.QPushButton('Export all', self)
        delete_button = QtWidgets.QPushButton('Delete', self)
        clear_button = QtWidgets.QPushButton('Clear', self)
        undo_button = QtWidgets.QPushButton('Undo', self)

        export_button.clicked.connect(self.export_image)
        export_all_button.clicked.connect(self.export_all)
        delete_button.clicked.connect(self.delete_image)
        clear_button.clicked.connect(self.clear_image)
        undo_button.clicked.connect(self.remove_last_point)

        self.left_button = QtWidgets.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        self.right_button = QtWidgets.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        self.up_button = QtWidgets.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        self.down_button = QtWidgets.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        self.px_shift_input = QtWidgets.QLineEdit('0', self)

        self.rot_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_right.png'), '', self)
        self.rot_counter_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_left.png'), '', self)
        self.rot_angle_input = QtWidgets.QLineEdit('0.0', self)

        self.px_shift_input.setFixedWidth(60)
        self.rot_angle_input.setFixedWidth(60)

        self.left_button.clicked.connect(self.move_left)
        self.right_button.clicked.connect(self.move_right)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)

        self.rot_counter_clockwise_button.clicked.connect(self.rot_left)
        self.rot_clockwise_button.clicked.connect(self.rot_right)

        self.left_button.setFixedWidth(60)
        self.right_button.setFixedWidth(60)
        self.up_button.setFixedWidth(60)
        self.down_button.setFixedWidth(60)
        self.rot_counter_clockwise_button.setFixedWidth(60)
        self.rot_clockwise_button.setFixedWidth(60)

        self.apply_button = QtWidgets.QPushButton('Apply changes', self)
        self.reset_button = QtWidgets.QPushButton('Reset', self)
        self.apply_button.clicked.connect(self.apply_changes)
        self.reset_button.clicked.connect(self.reset_changes)

        self.disable_manual_panel()

        self.manual_mode_checkbox = QtWidgets.QCheckBox('Manual mode', self)
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.clicked.connect(self.create_backup_image)

        grid_manual = QtWidgets.QGridLayout()
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

        self.shift_radio_button = QtWidgets.QRadioButton('Shift', self)
        self.rot_radio_button = QtWidgets.QRadioButton('Rot', self)
        self.shift_radio_button.setChecked(True)

        shift_rot_group = QtWidgets.QButtonGroup(self)
        shift_rot_group.addButton(self.shift_radio_button)
        shift_rot_group.addButton(self.rot_radio_button)

        alignButton = QtWidgets.QPushButton('Align', self)
        warpButton = QtWidgets.QPushButton('Warp', self)
        magnify_button = QtWidgets.QPushButton('Magnify', self)
        reshift_button = QtWidgets.QPushButton('Re-Shift', self)
        rerot_button = QtWidgets.QPushButton('Re-Rot', self)
        remag_button = QtWidgets.QPushButton('Re-Mag', self)
        rewarp_button = QtWidgets.QPushButton('Re-Warp', self)

        alignButton.clicked.connect(self.align_images)
        warpButton.clicked.connect(partial(self.warp_image, False))
        magnify_button.clicked.connect(self.magnify)
        reshift_button.clicked.connect(self.reshift)
        rerot_button.clicked.connect(self.rerotate)
        remag_button.clicked.connect(self.remagnify)
        rewarp_button.clicked.connect(self.rewarp)

        holo_no_ref_1_button = QtWidgets.QPushButton('FFT', self)
        holo_no_ref_2_button = QtWidgets.QPushButton('Holo', self)
        holo_with_ref_2_button = QtWidgets.QPushButton('Holo+Ref', self)
        holo_no_ref_3_button = QtWidgets.QPushButton('IFFT', self)

        holo_no_ref_1_button.clicked.connect(self.rec_holo_no_ref_1)
        holo_no_ref_2_button.clicked.connect(self.rec_holo_no_ref_2)
        holo_with_ref_2_button.clicked.connect(self.rec_holo_with_ref_2)
        holo_no_ref_3_button.clicked.connect(self.rec_holo_no_ref_3)

        hbox_holo = QtWidgets.QHBoxLayout()
        hbox_holo.addWidget(holo_no_ref_2_button)
        hbox_holo.addWidget(holo_with_ref_2_button)

        self.show_lines_checkbox = QtWidgets.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtWidgets.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.log_scale_checkbox = QtWidgets.QCheckBox('Log scale', self)
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_display)

        unwrap_button = QtWidgets.QPushButton('Unwrap', self)
        wrap_button = QtWidgets.QPushButton('Wrap', self)

        unwrap_button.clicked.connect(self.unwrap_img_phase)
        wrap_button.clicked.connect(self.wrap_img_phase)

        self.amp_radio_button = QtWidgets.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtWidgets.QRadioButton('Phase', self)
        self.cos_phs_radio_button = QtWidgets.QRadioButton('Phase cosine', self)
        self.amp_radio_button.setChecked(True)

        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)
        self.cos_phs_radio_button.toggled.connect(self.update_display)

        amp_phs_group = QtWidgets.QButtonGroup(self)
        amp_phs_group.addButton(self.amp_radio_button)
        amp_phs_group.addButton(self.phs_radio_button)
        amp_phs_group.addButton(self.cos_phs_radio_button)

        self.gray_radio_button = QtWidgets.QRadioButton('Grayscale', self)
        self.color_radio_button = QtWidgets.QRadioButton('Color', self)
        self.gray_radio_button.setChecked(True)

        self.gray_radio_button.toggled.connect(self.update_display)
        self.color_radio_button.toggled.connect(self.update_display)

        color_group = QtWidgets.QButtonGroup(self)
        color_group.addButton(self.gray_radio_button)
        color_group.addButton(self.color_radio_button)

        hbox_unwrap_gray = QtWidgets.QHBoxLayout()
        hbox_unwrap_gray.addWidget(unwrap_button)
        hbox_unwrap_gray.addWidget(self.gray_radio_button)

        hbox_wrap_color = QtWidgets.QHBoxLayout()
        hbox_wrap_color.addWidget(wrap_button)
        hbox_wrap_color.addWidget(self.color_radio_button)

        fname_label = QtWidgets.QLabel('File name', self)
        self.fname_input = QtWidgets.QLineEdit(self.display.image.name, self)
        self.fname_input.setFixedWidth(150)

        aperture_label = QtWidgets.QLabel('Aperture [px]', self)
        self.aperture_input = QtWidgets.QLineEdit(str(const.aperture), self)

        hann_win_label = QtWidgets.QLabel('Hann window [px]', self)
        self.hann_win_input = QtWidgets.QLineEdit(str(const.hann_win), self)

        sum_button = QtWidgets.QPushButton('Sum', self)
        diff_button = QtWidgets.QPushButton('Diff', self)

        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)

        amp_factor_label = QtWidgets.QLabel('Amp. factor', self)
        self.amp_factor_input = QtWidgets.QLineEdit('2.0', self)

        amplify_button = QtWidgets.QPushButton('Amplify', self)
        amplify_button.clicked.connect(self.amplify_phase)

        int_width_label = QtWidgets.QLabel('Profile width [px]', self)
        self.int_width_input = QtWidgets.QLineEdit('1', self)

        plot_button = QtWidgets.QPushButton('Plot profile', self)
        plot_button.clicked.connect(self.plot_profile)

        sample_thick_label = QtWidgets.QLabel('Sample thickness [nm]', self)
        self.sample_thick_input = QtWidgets.QLineEdit('30', self)

        calc_B_button = QtWidgets.QPushButton('Calculate B', self)
        calc_grad_button = QtWidgets.QPushButton('Calculate gradient', self)

        calc_B_button.clicked.connect(self.calc_magnetic_field)
        calc_grad_button.clicked.connect(self.calc_phase_gradient)

        threshold_label = QtWidgets.QLabel('Int. threshold [0-1]', self)
        self.threshold_input = QtWidgets.QLineEdit('0.9', self)

        filter_contours_button = QtWidgets.QPushButton('Filter contours', self)
        filter_contours_button.clicked.connect(self.filter_contours)

        norm_phase_button = QtWidgets.QPushButton('Normalize phase', self)
        norm_phase_button.clicked.connect(self.norm_phase)

        grid_nav = QtWidgets.QGridLayout()
        grid_nav.addWidget(prev_button, 0, 0)
        grid_nav.addWidget(next_button, 0, 1)
        grid_nav.addWidget(lswap_button, 1, 0)
        grid_nav.addWidget(rswap_button, 1, 1)
        grid_nav.addWidget(flip_button, 2, 0)
        grid_nav.addWidget(clear_button, 2, 1)
        grid_nav.addWidget(delete_button, 3, 1)
        grid_nav.addLayout(hbox_zoom, 3, 0)
        grid_nav.addLayout(hbox_name, 4, 0)
        grid_nav.addWidget(undo_button, 4, 1)

        grid_disp = QtWidgets.QGridLayout()
        grid_disp.setColumnStretch(0, 0)
        grid_disp.setColumnStretch(1, 0)
        grid_disp.setColumnStretch(2, 0)
        grid_disp.addWidget(self.show_lines_checkbox, 1, 0)
        grid_disp.addWidget(self.show_labels_checkbox, 2, 0)
        grid_disp.addWidget(self.log_scale_checkbox, 3, 0)
        grid_disp.addWidget(self.amp_radio_button, 1, 1)
        grid_disp.addWidget(self.phs_radio_button, 2, 1)
        grid_disp.addWidget(self.cos_phs_radio_button, 3, 1)
        grid_disp.addLayout(hbox_unwrap_gray, 1, 2)
        grid_disp.addLayout(hbox_wrap_color, 2, 2)
        grid_disp.addWidget(norm_phase_button, 3, 2)
        grid_disp.addWidget(fname_label, 0, 4)
        grid_disp.addWidget(self.fname_input, 1, 4)
        grid_disp.addWidget(export_button, 2, 4)
        grid_disp.addWidget(export_all_button, 3, 4)

        vbox_sh_rot_rb = QtWidgets.QVBoxLayout()
        vbox_sh_rot_rb.addWidget(self.shift_radio_button)
        vbox_sh_rot_rb.addWidget(self.rot_radio_button)

        vbox_sh_rot_bt = QtWidgets.QVBoxLayout()
        vbox_sh_rot_bt.addWidget(reshift_button)
        vbox_sh_rot_bt.addWidget(rerot_button)

        alignButton.setFixedHeight(50)

        grid_align = QtWidgets.QGridLayout()
        grid_align.setColumnStretch(1, 1)
        grid_align.setColumnStretch(2, 1)
        grid_align.addLayout(vbox_sh_rot_rb, 0, 0)
        grid_align.addWidget(alignButton, 0, 1)
        grid_align.addWidget(magnify_button, 1, 1)
        grid_align.addWidget(warpButton, 2, 1)
        grid_align.addLayout(vbox_sh_rot_bt, 0, 2)
        grid_align.addWidget(remag_button, 1, 2)
        grid_align.addWidget(rewarp_button, 2, 2)

        grid_holo = QtWidgets.QGridLayout()
        grid_holo.addWidget(aperture_label, 0, 0)
        grid_holo.addWidget(self.aperture_input, 1, 0)
        grid_holo.addWidget(hann_win_label, 0, 1)
        grid_holo.addWidget(self.hann_win_input, 1, 1)
        grid_holo.addWidget(holo_no_ref_1_button, 2, 0)
        grid_holo.addLayout(hbox_holo, 2, 1)
        grid_holo.addWidget(holo_no_ref_3_button, 3, 0)
        grid_holo.addWidget(sum_button, 3, 1)
        grid_holo.addWidget(diff_button, 3, 2)
        grid_holo.addWidget(amp_factor_label, 0, 2)
        grid_holo.addWidget(self.amp_factor_input, 1, 2)
        grid_holo.addWidget(amplify_button, 2, 2)

        grid_plot = QtWidgets.QGridLayout()
        grid_plot.addWidget(sample_thick_label, 0, 0)
        grid_plot.addWidget(self.sample_thick_input, 1, 0)
        grid_plot.addWidget(calc_grad_button, 2, 0)
        grid_plot.addWidget(calc_B_button, 3, 0)
        grid_plot.addWidget(int_width_label, 0, 1)
        grid_plot.addWidget(self.int_width_input, 1, 1)
        grid_plot.addWidget(plot_button, 2, 1)
        grid_plot.addWidget(threshold_label, 0, 2)
        grid_plot.addWidget(self.threshold_input, 1, 2)
        grid_plot.addWidget(filter_contours_button, 2, 2)
        # grid_plot.addWidget(norm_phase_button, 3, 1)

        vbox_panel = QtWidgets.QVBoxLayout()
        vbox_panel.addLayout(grid_nav)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_disp)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_manual)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_align)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_holo)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_plot)

        hbox_panel = QtWidgets.QHBoxLayout()
        hbox_panel.addWidget(self.display)
        hbox_panel.addLayout(vbox_panel)

        vbox_main = QtWidgets.QVBoxLayout()
        vbox_main.addLayout(hbox_panel)
        vbox_main.addWidget(self.plot_widget)
        self.setLayout(vbox_main)

        self.move(250, 5)
        self.setWindowTitle('Holo window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

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

    def go_to_prev_image(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        if self.display.image.prev is not None:
            self.name_input.setText(self.display.image.prev.name)
            self.fname_input.setText(self.display.image.prev.name)
            self.manual_mode_checkbox.setChecked(False)
            self.disable_manual_panel()
        self.display.changeImage(toNext=False, dispAmp=is_amp_checked, dispPhs=is_phs_checked,
                                 logScale=is_log_scale_checked, dispLabs=is_show_labels_checked, color=is_color_checked)

    def go_to_next_image(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        if self.display.image.next is not None:
            self.name_input.setText(self.display.image.next.name)
            self.fname_input.setText(self.display.image.next.name)
            self.manual_mode_checkbox.setChecked(False)
            self.disable_manual_panel()
        self.display.changeImage(toNext=True, dispAmp=is_amp_checked, dispPhs=is_phs_checked,
                                 logScale=is_log_scale_checked, dispLabs=is_show_labels_checked, color=is_color_checked)

    def flip_image_h(self):
        imsup.flip_image_h(self.display.image)
        self.display.setImage()

    def export_image(self):
        curr_num = self.display.image.numInSeries
        fname = self.fname_input.text()
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()

        log = True if self.log_scale_checkbox.isChecked() else False
        color = True if self.color_radio_button.isChecked() else False

        if fname == '':
            fname = 'amp{0}'.format(curr_num) if is_amp_checked else 'phs{0}'.format(curr_num)

        curr_img = self.display.image
        if is_amp_checked:
            imsup.SaveAmpImage(curr_img, '{0}.png'.format(fname), log, color)
        elif is_phs_checked:
            imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname), log, color)
        else:
            phs_tmp = np.copy(curr_img.amPh.ph)
            curr_img.amPh.ph = np.cos(phs_tmp)
            imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname), log, color)
            curr_img.amPh.ph = np.copy(phs_tmp)
        print('Saved image as "{0}.png"'.format(fname))

    def export_all(self):
        curr_img = imsup.GetFirstImage(self.display.image)

        while curr_img is not None:
            curr_num = curr_img.numInSeries
            fname = curr_img.name
            is_amp_checked = self.amp_radio_button.isChecked()

            if fname == '':
                fname = 'amp{0}'.format(curr_num) if is_amp_checked else 'phs{0}'.format(curr_num)

            if is_amp_checked:
                imsup.SaveAmpImage(curr_img, '{0}.png'.format(fname))
            else:
                imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname))

            print('Saved image as "{0}.png"'.format(fname))
            curr_img = curr_img.next

    def delete_image(self):
        curr_img = self.display.image
        if curr_img.prev is None and curr_img.next is None:
            return

        curr_idx = curr_img.numInSeries - 1
        first_img = imsup.GetFirstImage(curr_img)
        tmp_img_list = imsup.CreateImageListFromFirstImage(first_img)

        if curr_img.prev is not None:
            curr_img.prev.next = None
            self.go_to_prev_image()
        else:
            curr_img.next.prev = None
            self.go_to_next_image()
            if curr_idx == 0:
                self.display.image.numInSeries = 1

        del tmp_img_list[curr_idx]
        del self.display.pointSets[curr_idx]
        tmp_img_list.UpdateLinks()
        del curr_img

    def toggle_lines(self):
        self.display.show_lines = not self.display.show_lines
        self.display.repaint()

    def toggle_labels(self):
        self.display.show_labs = not self.display.show_labs
        if self.display.show_labs:
            self.display.show_labels()
        else:
            self.display.hide_labels()

    def update_display(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.setImage(dispAmp=is_amp_checked, dispPhs=is_phs_checked, logScale=is_log_scale_checked, color=is_color_checked)

    def unwrap_img_phase(self):
        curr_img = self.display.image
        new_phs = tr.unwrap_phase(curr_img.amPh.ph)
        curr_img.amPh.ph = np.copy(new_phs)
        self.update_display()

    def wrap_img_phase(self):
        curr_img = self.display.image
        uw_min = np.min(curr_img.amPh.ph)

        if uw_min > 0:
            uw_min = 0
        new_phs = (curr_img.amPh.ph - uw_min) % (2 * np.pi) - np.pi

        curr_img.amPh.ph = np.copy(new_phs)
        self.update_display()

    def norm_phase(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) == 0:
            print('Mark the reference point on the image')
            return
        pt_disp = self.display.pointSets[curr_idx][0]
        pt_real = CalcRealTLCoords(curr_img.width, pt_disp)

        first_img = imsup.GetFirstImage(curr_img)
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        for img in img_list:
            new_phs = norm_phase_to_pt(img.amPh.ph, pt_real)
            img.amPh.ph = np.copy(new_phs)
            img.update_cos_phase()
        self.update_display()
        print('All phases normalized')

    def zoom_n_fragments(self):
        curr_idx = self.display.image.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) < 2:
            return

        curr_img = self.display.image
        [pt1, pt2] = self.display.pointSets[curr_idx][:2]
        disp_crop_coords = pt1 + pt2
        real_crop_coords = imsup.MakeSquareCoords(CalcRealTLCoords(curr_img.width, disp_crop_coords))

        n_to_zoom = np.int(self.n_to_zoom_input.text())
        img_list = imsup.CreateImageListFromFirstImage(curr_img)
        img_list2 = img_list[:n_to_zoom]
        print(len(img_list2))
        idx1 = curr_img.numInSeries + n_to_zoom
        idx2 = idx1 + n_to_zoom
        for img, n in zip(img_list2, range(idx1, idx2)):
            frag = zoom_fragment(img, real_crop_coords)
            img_list.insert(n, frag)

        img_list.UpdateLinks()

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

    def create_backup_image(self):
        if self.manual_mode_checkbox.isChecked():
            if self.backup_image is None:
                self.backup_image = imsup.copy_am_ph_image(self.display.image)
            self.enable_manual_panel()
        else:
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
            shifted_img = cc.shift_am_ph_image(tmp, total_shift)
        else:
            shifted_img = cc.shift_am_ph_image(bckp, total_shift)

        curr.amPh.am = np.copy(shifted_img.amPh.am)
        curr.amPh.ph = np.copy(shifted_img.amPh.ph)
        curr.shift = total_shift
        self.display.setImage()

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

        if curr.shift != 0:
            tmp = cc.shift_am_ph_image(bckp, curr.shift)
            rotated_img = tr.RotateImageSki(tmp, total_rot)
        else:
            rotated_img = tr.RotateImageSki(bckp, total_rot)

        curr.amPh.am = np.copy(rotated_img.amPh.am)
        curr.amPh.ph = np.copy(rotated_img.amPh.ph)
        curr.rot = total_rot
        self.display.setImage()

    def repeat_prev_mods(self):
        curr = imsup.copy_am_ph_image(self.backup_image)
        for mod in self.changes_made:
            curr = modify_image(curr, mod[:2], bool(mod[2]))
        self.display.image = curr

    def apply_changes(self):
        self.backup_image = None

    def reset_changes(self):
        self.display.image = imsup.copy_am_ph_image(self.backup_image)
        self.backup_image = None
        # self.changes_made = []
        self.display.setImage()

    def align_images(self):
        if self.shift_radio_button.isChecked():
            self.align_shift()
        else:
            self.align_rot()

    def align_rot(self):
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

        img1Rc = cc.shift_am_ph_image(img1Pad, rcShift)
        img2Rc = cc.shift_am_ph_image(img2Pad, rcShift)

        img1Rc = imsup.create_imgbuf_from_img(img1Rc)
        img2Rc = imsup.create_imgbuf_from_img(img2Rc)

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

        img1RcPad.MoveToCPU()
        img2Rot.MoveToCPU()
        img1RcPad.UpdateBuffer()
        img2Rot.UpdateBuffer()

        mag_factor = curr_img.width / img1RcPad.width
        img1_mag = tr.RescaleImageSki(img1RcPad, mag_factor)
        img2_mag = tr.RescaleImageSki(img2Rot, mag_factor)

        self.insert_img_after_curr(img1_mag)
        self.insert_img_after_curr(img2_mag)

        print('Rotation complete!')

    def align_shift(self):
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

        shifted_img2 = cc.shift_am_ph_image(curr_img, shift_avg)
        shifted_img2 = imsup.create_imgbuf_from_img(shifted_img2)
        self.insert_img_after_curr(shifted_img2)

    def reshift(self):
        curr_img = self.display.image
        shift = self.shift

        if self.shift_radio_button.isChecked():
            shifted_img = cc.shift_am_ph_image(curr_img, shift)
            shifted_img = imsup.create_imgbuf_from_img(shifted_img)
            self.insert_img_after_curr(shifted_img)
        else:
            bufSz = max([abs(x) for x in shift])
            dirs = 'tblr'
            padded_img = imsup.PadImage(curr_img, bufSz, 0.0, dirs)
            shifted_img = cc.shift_am_ph_image(padded_img, shift)
            shifted_img = imsup.create_imgbuf_from_img(shifted_img)

            resc_factor = curr_img.width / padded_img.width
            resc_img = tr.RescaleImageSki(shifted_img, resc_factor)
            self.insert_img_after_curr(resc_img)

    def rerotate(self):
        curr_img = self.display.image
        rot_angle = self.rot_angle
        rotated_img = tr.RotateImageSki(curr_img, rot_angle)
        self.insert_img_after_curr(rotated_img)

    def magnify(self):
        curr_img = self.display.image
        ref_img = curr_img.prev
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points1 = self.display.pointSets[curr_idx - 1]
        points2 = self.display.pointSets[curr_idx]
        n_points1 = len(points1)
        n_points2 = len(points2)

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

        mags = [dist1 / dist2 for dist1, dist2 in zip(poly1_dists, poly2_dists)]
        mag_avg = np.average(mags)
        self.mag_coeff = mag_avg

        print('---- Magnification ----')
        print(['mag{0} = {1:.2f}x\n'.format(idx + 1, mag) for idx, mag in zip(range(len(mags)), mags)])
        print('------------------')
        print('Average magnification = {0:.2f}x'.format(mag_avg))

        magnified_img = tr.RescaleImageSki(curr_img, mag_avg)

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

        self.insert_img_after_curr(resc_img1)
        self.insert_img_after_curr(resc_img2)

        print('Magnification complete!')

    def remagnify(self):
        curr_img = self.display.image
        mag_factor = self.mag_coeff

        mag_img = tr.RescaleImageSki(curr_img, mag_factor)
        pad_sz = (mag_img.width - curr_img.width) // 2

        if pad_sz > 0:
            pad_img = imsup.pad_img_from_ref(curr_img, mag_img.width, 0.0, 'tblr')
            resc_factor = curr_img.width / pad_img.width
            resc_img = tr.RescaleImageSki(pad_img, resc_factor)
        else:
            resc_img = imsup.pad_img_from_ref(mag_img, curr_img.width, 0.0, 'tblr')

        self.insert_img_after_curr(resc_img)

    def warp_image(self, more_accurate=False):
        curr_img = self.display.image
        curr_idx = self.display.image.numInSeries - 1
        real_points1 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx-1])
        real_points2 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx])
        user_points1 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points1)
        user_points2 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points2)

        self.warp_points = [ user_points1, user_points2 ]

        if more_accurate:
            n_div = const.nDivForUnwarp
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

    def rec_holo_no_ref_1(self):
        holo_img = self.display.image
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        self.insert_img_after_curr(holo_fft)

    def rec_holo_no_ref_2(self):
        holo_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[holo_fft.numInSeries-1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(holo_fft.width, dpts)
        rpt1 = rpts[:2]     # konwencja x, y
        rpt2 = rpts[2:]

        sband = np.copy(holo_fft.amPh.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])     # konwencja y, x
        sband_xy = holo.find_img_max(sband)     # konwencja y, x
        sband_xy.reverse()

        sband_xy = [ px + sx for px, sx in zip(rpt1, sband_xy) ]    # konwencja x, y

        mid = holo_fft.width // 2
        shift = [ mid - sband_xy[1], mid - sband_xy[0] ]    # konwencja x, y

        aperture = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        sband_img_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_sz=aperture, N_hann=hann_window)
        self.log_scale_checkbox.setChecked(False)
        self.insert_img_after_curr(sband_img_ap)

    def rec_holo_with_ref_2(self):
        ref_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[ref_fft.numInSeries - 1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(ref_fft.width, dpts)
        rpt1 = rpts[:2]  # konwencja x, y
        rpt2 = rpts[2:]

        sband = np.copy(ref_fft.amPh.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])  # konwencja y, x
        sband_xy = holo.find_img_max(sband)  # konwencja y, x
        sband_xy.reverse()

        sband_xy = [px + sx for px, sx in zip(rpt1, sband_xy)]  # konwencja x, y

        mid = ref_fft.width // 2
        shift = [mid - sband_xy[1], mid - sband_xy[0]]  # konwencja x, y

        aperture = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        ref_sband_ap = holo.rec_holo_no_ref_2(ref_fft, shift, ap_sz=aperture, N_hann=hann_window)

        holo_img = self.display.image.next
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        holo_sband_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_sz=aperture, N_hann=hann_window)

        self.log_scale_checkbox.setChecked(False)
        self.insert_img_after_curr(ref_sband_ap)
        self.insert_img_after_curr(holo_sband_ap)

    def rec_holo_no_ref_3(self):
        sband_img = self.display.image
        rec_holo = holo.rec_holo_no_ref_3(sband_img)
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
        self.insert_img_after_curr(phs_sum)

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)
        np.savetxt('ph_diff', phs_diff.amPh.ph)     # !!!
        self.insert_img_after_curr(phs_diff)

    def amplify_phase(self):
        curr_img = self.display.image
        curr_name = self.name_input.text()
        amp_factor = float(self.amp_factor_input.text())

        phs_amplified = imsup.copy_am_ph_image(curr_img)
        phs_amplified.amPh.ph *= amp_factor
        phs_amplified.update_cos_phase()
        phs_amplified.name = '{0}_x{1:.0f}'.format(curr_name, amp_factor)
        self.insert_img_after_curr(phs_amplified)
        self.cos_phs_radio_button.setChecked(True)

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
        img_shifted = cc.shift_am_ph_image(curr_img, shift_to_rot_center)

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
        img_cropped = imsup.crop_am_ph_roi_cpu(img_rot, frag_coords)

        # calculate projection of intensity
        if self.amp_radio_button.isChecked():
            int_matrix = np.copy(img_cropped.amPh.am)
        elif self.phs_radio_button.isChecked():
            ph_min = np.min(img_cropped.amPh.ph)
            ph_fix = -ph_min if ph_min < 0 else 0
            img_cropped.amPh.ph += ph_fix
            int_matrix = np.copy(img_cropped.amPh.ph)
        else:
            cos_ph_min = np.min(img_cropped.cos_phase)
            cos_ph_fix = -cos_ph_min if cos_ph_min < 0 else 0
            img_cropped.cos_phase += cos_ph_fix
            int_matrix = np.copy(img_cropped.cos_phase)
        print(int_matrix)
        int_profile = np.sum(int_matrix, proj_dir)  # 0 - horizontal projection, 1 - vertical projection
        dists = np.arange(0, int_profile.shape[0], 1) * px_sz
        dists *= 1e9
        print('wat2')

        self.plot_widget.plot(dists, int_profile, 'Distance [nm]', 'Intensity [a.u.]')

    def calc_phase_gradient(self):
        curr_img = self.display.image
        dx_img = imsup.copy_am_ph_image(curr_img)
        dy_img = imsup.copy_am_ph_image(curr_img)
        grad_img = imsup.copy_am_ph_image(curr_img)
        dx, dy = np.gradient(curr_img.amPh.ph)
        dr = np.sqrt(dx * dx + dy * dy)
        dphi = np.arctan2(dy, dx)
        dx_img.amPh.ph = np.copy(dx)
        dy_img.amPh.ph = np.copy(dy)
        grad_img.amPh.am = np.copy(dr)
        grad_img.amPh.ph = np.copy(dphi)
        self.insert_img_after_curr(dx_img)
        self.insert_img_after_curr(dy_img)
        self.insert_img_after_curr(grad_img)

    def calc_magnetic_field(self):
        pt1, pt2 = self.plot_widget.markedPointsData
        d_dist = np.abs(pt1[0] - pt2[0]) * 1e-9
        d_phase = np.abs(pt1[1] - pt2[1])
        print('{0:.2f} rad'.format(d_phase))
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_in_plane = (const.dirac_const / sample_thickness) * (d_phase / d_dist)
        print('B = {0:.2f} T'.format(B_in_plane))

    def filter_contours(self):
        curr_img = self.display.image
        conts = np.copy(curr_img.cos_phase)
        conts_scaled = imsup.ScaleImage(conts, 0, 1)
        threshold = float(self.threshold_input.text())
        conts_scaled[conts_scaled < threshold] = 0
        img_filtered = imsup.copy_am_ph_image(curr_img)
        img_filtered.cos_phase = np.copy(conts_scaled)
        self.insert_img_after_curr(img_filtered)
        # find_contours(self.display.image)

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

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData, pxDims = dm3.ReadDm3File(imgPath)
        imsup.Image.px_dim_default = pxDims[0]
        imgData = np.abs(imgData)
        img = imsup.ImageExp(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'],
                             num=imgNum, px_dim_sz=pxDims[0])
        # img.LoadAmpData(np.sqrt(imgData).astype(np.float32))
        img.LoadAmpData(imgData.astype(np.float32))
        # img.amPh.ph = np.copy(img.amPh.am)
        # ---
        # imsup.RemovePixelArtifacts(img, const.minPxThreshold, const.maxPxThreshold)
        # imsup.RemovePixelArtifacts(img, 0.7, 1.3)
        # img.UpdateBuffer()
        # ---
        imgList.append(img)

        imgNum += 1
        imgNumTextNew = imgNumText.replace(str(imgNum-1), str(imgNum))
        if imgNum == 10:
            imgNumTextNew = imgNumTextNew[1:]
        imgPath = RReplace(imgPath, imgNumText, imgNumTextNew, 1)
        imgNumText = imgNumTextNew

    imgList.UpdateLinks()
    return imgList[0]

# --------------------------------------------------------

def zoom_fragment(img, coords):
    crop_img = imsup.crop_am_ph_roi(img, coords)
    crop_img = imsup.create_imgbuf_from_img(crop_img)
    crop_img.MoveToCPU()

    orig_width = img.width
    crop_width = np.abs(coords[2] - coords[0])
    zoom_factor = orig_width / crop_width
    zoom_img = tr.RescaleImageSki(crop_img, zoom_factor)
    zoom_img.px_dim *= zoom_factor
    # self.insert_img_after_curr(zoom_img)
    return zoom_img

# --------------------------------------------------------

def modify_image(img, mod=list([0, 0]), is_shift=True):
    if is_shift:
        mod_img = cc.shift_am_ph_image(img, mod)
    else:
        mod_img = tr.RotateImageSki(img, mod[0])

    return mod_img

# --------------------------------------------------------

def norm_phase_to_pt(phase, pt):
    y, x = pt
    phase_norm = phase - phase[y, x]
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
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int(dc * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealTLCoordsForSetOfPoints(imgWidth, points):
    realCoords = [ CalcRealTLCoords(imgWidth, pt) for pt in points ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int((dc - dispWidth // 2) * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoordsForSetOfPoints(imgWidth, points):
    realPoints = [ CalcRealCoords(imgWidth, pt) for pt in points ]
    return realPoints

# --------------------------------------------------------

def CalcRealTLCoordsForPaddedImage(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
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
    dispCoords = [ (rc * factor) + const.ccWidgetDim // 2 for rc in realCoords ]
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

def SwitchXY(xy):
    return [xy[1], xy[0]]

# --------------------------------------------------------

def RReplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def RunTriangulationWindow():
    app = QtWidgets.QApplication(sys.argv)
    trWindow = TriangulateWidget()
    sys.exit(app.exec_())

# --------------------------------------------------------

def trace_contour(arr, xy):
    contour = []
    x, y = xy
    adj_sum = 1
    mid = np.ones(2)
    last_xy = [1, 1]
    print(xy)

    while adj_sum > 0:
        adj_arr = arr[x-1:x+2, y-1:y+2]
        adj_arr[last_xy[0], last_xy[1]] = 0
        adj_arr[1, 1] = 0
        adj_sum = np.sum(np.array(adj_arr))
        if adj_sum > 0:
            next_xy = [ idx[0] for idx in np.where(adj_arr == 1) ]
            last_xy = list(2 * mid - np.array(next_xy))
            next_xy = list(np.array(next_xy)-1 + np.array(xy))
            contour.append(next_xy)
            x, y = next_xy
            xy = [x, y]
            print(next_xy)

    print(len(contour))
    cont_arr = np.zeros(arr.shape)
    for idxs in contour:
        cont_arr[idxs[0], idxs[1]] = 1

    # cont_img = imsup.ImageExp(cont_arr.shape[0], cont_arr.shape[1])
    # cont_img.LoadAmpData(cont_arr)
    # imsup.DisplayAmpImage(cont_img)

    return contour

# --------------------------------------------------------

def find_contours(img):
    # arrow_dirs = np.zeros((img.height, img.width))
    ph_arr = np.copy(img.amPh.ph)
    ph_arr_scaled = imsup.ScaleImage(ph_arr, 0, 1)
    ph_arr_scaled[ph_arr_scaled < 0.98] = 0
    ph_arr_scaled[ph_arr_scaled >= 0.98] = 1
    # ph_arr_corr = imsup.ScaleImage(ph_arr_scaled, 0, 1)

    for i in range(100, img.height):
        for j in range(100, img.width):
            if ph_arr_scaled[i, j] == 1:
                print('Found one!')
                print(i, j)
                contour = trace_contour(ph_arr_scaled, [i, j])
                return contour