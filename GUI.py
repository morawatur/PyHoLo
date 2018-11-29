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

#-------------------------------------------------------------------

def read_dm3_file(fpath):
    img_data, px_dims = dm3.ReadDm3File(fpath)
    imsup.Image.px_dim_default = px_dims[0]

    holo_img = imsup.ImageExp(img_data.shape[0], img_data.shape[1])
    holo_img.LoadAmpData(np.sqrt(img_data).astype(np.float32))

    return holo_img

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

class LabelExt(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        self.image = image
        self.setImage()
        self.pointSets = [[]]
        self.show_lines = True
        self.show_labs = True
        self.rgb_cm = RgbColorTable()

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
        currPos = [pos.x(), pos.y()]
        self.pointSets[self.image.numInSeries - 1].append(currPos)
        self.repaint()

        if self.parent().show_labels_checkbox.isChecked():
            lab = QtWidgets.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries - 1])), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pos.x()+4, pos.y()+4)
            lab.show()

    def setImage(self, dispAmp=True, dispPhs=False, logScale=False, color=False, update_bcg=False, bright=0, cont=255, gamma=1.0):
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

        if not update_bcg:
            # buf_scaled = imsup.ScaleImage(self.image.buffer, 0.0, 255.0)
            buf_scaled = update_image_bright_cont_gamma(self.image.buffer, brg=self.image.bias, cnt=self.image.gain, gam=self.image.gamma)
        else:
            self.image.bias = bright
            self.image.gain = cont
            self.image.gamma = gamma
            buf_scaled = update_image_bright_cont_gamma(self.image.buffer, brg=bright, cnt=cont, gam=gamma)

        # final image with all properties set
        q_image = QtGui.QImage(buf_scaled.astype(np.uint8), self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)
        if color:
            q_image.setColorTable(self.rgb_cm.cm)

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.disp_dim)
        self.setPixmap(pixmap)
        self.repaint()

    def update_labs(self, dispLabs=True):
        if len(self.pointSets) < self.image.numInSeries:
            self.pointSets.append([])

        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

        if dispLabs:
            self.show_labels()

    def change_image(self, new_idx, dispAmp=True, dispPhs=False, logScale=False, dispLabs=True, color=False):
        curr = self.image
        first = imsup.GetFirstImage(curr)
        imgs = imsup.CreateImageListFromFirstImage(first)
        if 0 > new_idx > len(imgs) - 1:
            return

        new_img = imgs[new_idx]
        new_img.ReIm2AmPh()
        self.image = new_img
        self.setImage(dispAmp, dispPhs, logScale, color)
        self.update_labs(dispLabs)

    def change_image_adjacent(self, dir_to_img=1, dispAmp=True, dispPhs=False, logScale=False, dispLabs=True,
                              color=False):
        if dir_to_img == 1:
            new_img = self.image.next
        else:
            new_img = self.image.prev

        if new_img is None:
            return

        new_img.ReIm2AmPh()
        self.image = new_img
        self.setImage(dispAmp, dispPhs, logScale, color)
        self.update_labs(dispLabs)

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
        self.display.setFixedWidth(const.disp_dim)
        self.display.setFixedHeight(const.disp_dim)
        self.plot_widget = PlotWidget()
        self.backup_image = None
        self.changes_made = []
        self.shift = [0, 0]
        self.rot_angle = 0
        self.mag_coeff = 1.0
        self.warp_points = []
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(350)

        # ------------------------------
        # Navigation panel (1)
        # ------------------------------

        self.clear_prev_checkbox = QtWidgets.QCheckBox('Clear prev. images', self)
        self.clear_prev_checkbox.setChecked(False)

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

        self.name_input = QtWidgets.QLineEdit('self.display.image.name', self)
        self.n_to_zoom_input = QtWidgets.QLineEdit('1', self)

        hbox_name = QtWidgets.QHBoxLayout()
        hbox_name.addWidget(set_name_button)
        hbox_name.addWidget(self.name_input)

        hbox_zoom = QtWidgets.QHBoxLayout()
        hbox_zoom.addWidget(zoom_button)
        hbox_zoom.addWidget(self.n_to_zoom_input)

        self.tab_nav = QtWidgets.QWidget()
        self.tab_nav.layout = QtWidgets.QGridLayout()
        self.tab_nav.layout.setColumnStretch(0, 1)
        self.tab_nav.layout.setColumnStretch(1, 1)
        self.tab_nav.layout.setColumnStretch(2, 1)
        self.tab_nav.layout.setColumnStretch(3, 1)
        self.tab_nav.layout.setColumnStretch(4, 1)
        self.tab_nav.layout.setColumnStretch(5, 1)
        self.tab_nav.layout.setRowStretch(0, 1)
        self.tab_nav.layout.setRowStretch(7, 1)
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
        self.tab_nav.setLayout(self.tab_nav.layout)

        # ------------------------------
        # Display panel (2)
        # ------------------------------

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

        fname_label = QtWidgets.QLabel('File name', self)
        self.fname_input = QtWidgets.QLineEdit(self.display.image.name, self)

        unwrap_button = QtWidgets.QPushButton('Unwrap', self)
        wrap_button = QtWidgets.QPushButton('Wrap', self)
        export_button = QtWidgets.QPushButton('Export', self)
        export_all_button = QtWidgets.QPushButton('Export all', self)
        norm_phase_button = QtWidgets.QPushButton('Normalize phase', self)

        self.export_png_radio_button = QtWidgets.QRadioButton('PNG image', self)
        self.export_bin_radio_button = QtWidgets.QRadioButton('Binary', self)
        self.export_png_radio_button.setChecked(True)

        export_group = QtWidgets.QButtonGroup(self)
        export_group.addButton(self.export_png_radio_button)
        export_group.addButton(self.export_bin_radio_button)

        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)
        self.cos_phs_radio_button.toggled.connect(self.update_display)
        self.gray_radio_button.toggled.connect(self.update_display)
        self.color_radio_button.toggled.connect(self.update_display)
        unwrap_button.clicked.connect(self.unwrap_img_phase)
        wrap_button.clicked.connect(self.wrap_img_phase)
        export_button.clicked.connect(self.export_image)
        export_all_button.clicked.connect(self.export_all)
        norm_phase_button.clicked.connect(self.norm_phase)

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
        grid_disp.addWidget(norm_phase_button, 3, 3, 1, 2)

        grid_exp = QtWidgets.QGridLayout()
        grid_exp.setColumnStretch(0, 1)
        grid_exp.setColumnStretch(4, 1)
        grid_exp.setRowStretch(0, 1)
        grid_exp.setRowStretch(3, 1)
        grid_exp.addWidget(fname_label, 1, 1)
        grid_exp.addWidget(self.fname_input, 2, 1)
        grid_exp.addWidget(export_button, 1, 2)
        grid_exp.addWidget(export_all_button, 2, 2)
        grid_exp.addWidget(self.export_png_radio_button, 1, 3)
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

        self.px_shift_input = QtWidgets.QLineEdit('0', self)
        self.rot_angle_input = QtWidgets.QLineEdit('0.0', self)

        self.manual_mode_checkbox = QtWidgets.QCheckBox('Manual mode', self)
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.clicked.connect(self.create_backup_image)

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
        magnify_button = QtWidgets.QPushButton('Magnify', self)
        reshift_button = QtWidgets.QPushButton('Re-Shift', self)
        rerot_button = QtWidgets.QPushButton('Re-Rot', self)
        remag_button = QtWidgets.QPushButton('Re-Mag', self)
        rewarp_button = QtWidgets.QPushButton('Re-Warp', self)
        cross_corr_w_prev_button = QtWidgets.QPushButton('Cross corr. w. prev.', self)
        cross_corr_all_button = QtWidgets.QPushButton('Cross corr. all', self)

        auto_shift_button.clicked.connect(self.auto_shift_image)
        auto_rot_button.clicked.connect(self.auto_rot_image)
        warpButton.clicked.connect(partial(self.warp_image, False))
        magnify_button.clicked.connect(self.magnify)
        reshift_button.clicked.connect(self.reshift)
        rerot_button.clicked.connect(self.rerotate)
        remag_button.clicked.connect(self.remagnify)
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
        grid_auto.setRowStretch(5, 1)
        grid_auto.addWidget(auto_shift_button, 1, 1)
        grid_auto.addWidget(auto_rot_button, 2, 1)
        grid_auto.addWidget(magnify_button, 3, 1)
        grid_auto.addWidget(warpButton, 4, 1)
        grid_auto.addWidget(reshift_button, 1, 2)
        grid_auto.addWidget(rerot_button, 2, 2)
        grid_auto.addWidget(remag_button, 3, 2)
        grid_auto.addWidget(rewarp_button, 4, 2)
        grid_auto.addWidget(cross_corr_w_prev_button, 1, 3)
        grid_auto.addWidget(cross_corr_all_button, 2, 3)

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
        rm_gradient_button = QtWidgets.QPushButton('Remove gradient', self)

        aperture_label = QtWidgets.QLabel('Aperture [px]', self)
        self.aperture_input = QtWidgets.QLineEdit(str(const.aperture), self)

        hann_win_label = QtWidgets.QLabel('Hann window [px]', self)
        self.hann_win_input = QtWidgets.QLineEdit(str(const.hann_win), self)

        amp_factor_label = QtWidgets.QLabel('Amp. factor', self)
        self.amp_factor_input = QtWidgets.QLineEdit('2.0', self)

        holo_no_ref_1_button.clicked.connect(self.rec_holo_no_ref_1)
        holo_no_ref_2_button.clicked.connect(self.rec_holo_no_ref_2)
        holo_with_ref_2_button.clicked.connect(self.rec_holo_with_ref_2)
        holo_no_ref_3_button.clicked.connect(self.rec_holo_no_ref_3)
        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)
        amplify_button.clicked.connect(self.amplify_phase)
        rm_gradient_button.clicked.connect(self.remove_gradient)

        hbox_holo = QtWidgets.QHBoxLayout()
        hbox_holo.addWidget(holo_no_ref_2_button)
        hbox_holo.addWidget(holo_with_ref_2_button)

        self.tab_holo = QtWidgets.QWidget()
        self.tab_holo.layout = QtWidgets.QGridLayout()
        self.tab_holo.layout.setColumnStretch(0, 1)
        self.tab_holo.layout.setColumnStretch(4, 1)
        self.tab_holo.layout.setRowStretch(0, 1)
        self.tab_holo.layout.setRowStretch(6, 1)
        self.tab_holo.layout.addWidget(aperture_label, 1, 1)
        self.tab_holo.layout.addWidget(self.aperture_input, 2, 1)
        self.tab_holo.layout.addWidget(hann_win_label, 1, 2)
        self.tab_holo.layout.addWidget(self.hann_win_input, 2, 2)
        self.tab_holo.layout.addWidget(holo_no_ref_1_button, 3, 1)
        self.tab_holo.layout.addLayout(hbox_holo, 3, 2)
        self.tab_holo.layout.addWidget(holo_no_ref_3_button, 4, 1)
        self.tab_holo.layout.addWidget(sum_button, 4, 2)
        self.tab_holo.layout.addWidget(diff_button, 4, 3)
        self.tab_holo.layout.addWidget(amp_factor_label, 1, 3)
        self.tab_holo.layout.addWidget(self.amp_factor_input, 2, 3)
        self.tab_holo.layout.addWidget(amplify_button, 3, 3)
        self.tab_holo.layout.addWidget(rm_gradient_button, 5, 1)
        self.tab_holo.setLayout(self.tab_holo.layout)

        # ------------------------------
        # Magnetic calculations panel (6)
        # ------------------------------

        plot_button = QtWidgets.QPushButton('Plot profile', self)
        calc_B_button = QtWidgets.QPushButton('Calculate B', self)
        calc_grad_button = QtWidgets.QPushButton('Calculate gradient', self)
        filter_contours_button = QtWidgets.QPushButton('Filter contours', self)

        int_width_label = QtWidgets.QLabel('Profile width [px]', self)
        self.int_width_input = QtWidgets.QLineEdit('1', self)

        sample_thick_label = QtWidgets.QLabel('Sample thickness [nm]', self)
        self.sample_thick_input = QtWidgets.QLineEdit('30', self)

        threshold_label = QtWidgets.QLabel('Int. threshold [0-1]', self)
        self.threshold_input = QtWidgets.QLineEdit('0.9', self)

        plot_button.clicked.connect(self.plot_profile)
        calc_B_button.clicked.connect(self.calc_magnetic_field)
        calc_grad_button.clicked.connect(self.calc_phase_gradient)
        filter_contours_button.clicked.connect(self.filter_contours)

        self.tab_calc = QtWidgets.QWidget()
        self.tab_calc.layout = QtWidgets.QGridLayout()
        self.tab_calc.layout.setColumnStretch(0, 1)
        self.tab_calc.layout.setColumnStretch(4, 1)
        self.tab_calc.layout.setRowStretch(0, 1)
        self.tab_calc.layout.setRowStretch(5, 1)
        self.tab_calc.layout.addWidget(sample_thick_label, 1, 1)
        self.tab_calc.layout.addWidget(self.sample_thick_input, 2, 1)
        self.tab_calc.layout.addWidget(calc_grad_button, 3, 1)
        self.tab_calc.layout.addWidget(calc_B_button, 4, 1)
        self.tab_calc.layout.addWidget(int_width_label, 1, 2)
        self.tab_calc.layout.addWidget(self.int_width_input, 2, 2)
        self.tab_calc.layout.addWidget(plot_button, 3, 2)
        self.tab_calc.layout.addWidget(threshold_label, 1, 3)
        self.tab_calc.layout.addWidget(self.threshold_input, 2, 3)
        self.tab_calc.layout.addWidget(filter_contours_button, 3, 3)
        self.tab_calc.setLayout(self.tab_calc.layout)

        # ------------------------------
        # Bright/Gamma/Contrast panel (7)
        # ------------------------------

        bright_label = QtWidgets.QLabel('Brightness', self)
        cont_label = QtWidgets.QLabel('Contrast', self)
        gamma_label = QtWidgets.QLabel('Gamma', self)

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

        self.bright_input = QtWidgets.QLineEdit('0', self)
        self.cont_input = QtWidgets.QLineEdit('255', self)
        self.gamma_input = QtWidgets.QLineEdit('1.0', self)

        reset_bright_button = QtWidgets.QPushButton('Reset B', self)
        reset_cont_button = QtWidgets.QPushButton('Reset C', self)
        reset_gamma_button = QtWidgets.QPushButton('Reset G', self)

        self.bright_slider.valueChanged.connect(self.disp_bright_value)
        self.cont_slider.valueChanged.connect(self.disp_cont_value)
        self.gamma_slider.valueChanged.connect(self.disp_gamma_value)

        self.bright_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.cont_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.gamma_slider.sliderReleased.connect(self.update_display_and_bcg)

        self.bright_input.returnPressed.connect(self.update_display_and_bcg)
        self.cont_input.returnPressed.connect(self.update_display_and_bcg)
        self.gamma_input.returnPressed.connect(self.update_display_and_bcg)

        reset_bright_button.clicked.connect(self.reset_bright)
        reset_cont_button.clicked.connect(self.reset_cont)
        reset_gamma_button.clicked.connect(self.reset_gamma)

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
        vbox_panel.addWidget(self.tabs)
        vbox_panel.addWidget(self.plot_widget)

        hbox_main = QtWidgets.QHBoxLayout()
        hbox_main.addWidget(self.display)
        hbox_main.addLayout(vbox_panel)

        self.setLayout(hbox_main)

        self.reset_image_names()  # !!!

        self.move(250, 50)
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

    def reset_image_names(self):
        curr_img = self.display.image
        first_img = imsup.GetFirstImage(curr_img)
        img_queue = imsup.CreateImageListFromFirstImage(first_img)
        for img, idx in zip(img_queue, range(len(img_queue))):
            img.numInSeries = idx + 1
            img.name = 'img0{0}'.format(idx + 1) if idx < 9 else 'img{0}'.format(idx + 1)
        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)

    def go_to_image(self, new_idx):
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
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
        self.display.update_labs(is_show_labels_checked)
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
        imsup.flip_image_h(self.display.image)
        self.display.setImage()

    def save_amp_as_png(self, fname, log, color):
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
        amp_to_save.save('{0}.png'.format(fname))

    def save_phs_as_png(self, fname, log, color):
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
        phs_to_save.save('{0}.png'.format(fname))

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

        if self.export_bin_radio_button.isChecked():
            fname_ext = ''
            if is_amp_checked:
                # np.save(fname, curr_img.amPh.am)
                curr_img.amPh.am.tofile(fname)
            elif is_phs_checked:
                # np.save(fname, curr_img.amPh.ph)
                curr_img.amPh.ph.tofile(fname)
            else:
                cos_phs = np.cos(curr_img.amPh.ph)
                cos_phs.tofile(fname)
                # np.save(fname, cos_phs)
            print('Saved image to binary file: "{0}"'.format(fname))
        else:
            fname_ext = '.png'
            log = True if self.log_scale_checkbox.isChecked() else False
            color = True if self.color_radio_button.isChecked() else False

            if is_amp_checked:
                # imsup.SaveAmpImage(curr_img, '{0}.png'.format(fname), True, log, color)
                self.save_amp_as_png(fname, log, color)
            elif is_phs_checked:
                # imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname), True, log, color)
                self.save_phs_as_png(fname, log, color)
            else:
                phs_tmp = np.copy(curr_img.amPh.ph)
                curr_img.amPh.ph = np.cos(phs_tmp)
                # imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname), True log, color)
                self.save_phs_as_png(fname, log, color)
                curr_img.amPh.ph = np.copy(phs_tmp)
            print('Saved image as "{0}.png"'.format(fname))

        # save log file
        log_fname = '{0}_log.txt'.format(fname)
        with open(log_fname, 'w') as log_file:
            log_file.write('File name:\t{0}{1}\n'
                           'Image name:\t{2}\n'
                           'Image size:\t{3}x{4}\n'
                           'Calibration:\t{5} nm\n'.format(fname, fname_ext, curr_img.name, curr_img.width,
                                                           curr_img.height, curr_img.px_dim * 1e9))
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
            print('You have to mark two points on the image in order to zoom!')
            return

        curr_img = self.display.image
        pt1, pt2 = self.display.pointSets[curr_idx][:2]
        pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
        disp_crop_coords = pt1 + pt2
        real_tl_coords = CalcRealTLCoords(curr_img.width, disp_crop_coords)
        real_sq_coords = imsup.MakeSquareCoords(real_tl_coords)

        n_to_zoom = np.int(self.n_to_zoom_input.text())
        first_img = imsup.GetFirstImage(curr_img)
        insert_idx = curr_idx + n_to_zoom
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        img_list2 = img_list[curr_idx:insert_idx]

        for img, n in zip(img_list2, range(insert_idx, insert_idx + n_to_zoom)):
            frag = zoom_fragment(img, real_sq_coords)
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
            shifted_img = imsup.shift_am_ph_image(tmp, total_shift)
        else:
            shifted_img = imsup.shift_am_ph_image(bckp, total_shift)

        curr.amPh.am = np.copy(shifted_img.amPh.am)
        curr.amPh.ph = np.copy(shifted_img.amPh.ph)
        curr.shift = total_shift

        is_amp = self.amp_radio_button.isChecked()
        is_phs = self.phs_radio_button.isChecked()
        is_log = self.log_scale_checkbox.isChecked()
        is_col = self.color_radio_button.isChecked()

        self.display.setImage(dispAmp=is_amp, dispPhs=is_phs, logScale=is_log, color=is_col, update_bcg=True)

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
            tmp = imsup.shift_am_ph_image(bckp, curr.shift)
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
        self.insert_img_after_curr(shifted_img2)

    def reshift(self):
        curr_img = self.display.image
        shift = self.shift
        shifted_img = imsup.shift_am_ph_image(curr_img, shift)
        self.insert_img_after_curr(shifted_img)

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

    def rec_holo_no_ref_1(self):
        holo_img = self.display.image
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        self.log_scale_checkbox.setChecked(True)
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
        self.log_scale_checkbox.setChecked(True)
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
        self.insert_img_after_curr(phs_sum)

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)
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

    def remove_gradient(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        p1, p2, p3 = self.display.pointSets[curr_idx][:3]
        p1.append(curr_img.amPh.ph[p1[1], p1[0]])
        p2.append(curr_img.amPh.ph[p2[1], p2[0]])
        p3.append(curr_img.amPh.ph[p3[1], p3[0]])
        grad_plane = tr.Plane(0, 0, 0)
        grad_plane.getFromThreePoints(p1, p2, p3)
        print(grad_plane.a, grad_plane.b, grad_plane.c)
        grad_arr = grad_plane.fillPlane(curr_img.height, curr_img.width)
        grad_img = imsup.ImageExp(curr_img.height, curr_img.width)
        grad_img.amPh.ph = np.copy(grad_arr)
        print(grad_arr[p1[1], p1[0]])
        print(grad_arr[p2[1], p2[0]])
        print(grad_arr[p3[1], p3[0]])
        print(p1[2], p2[2], p3[2])
        self.insert_img_after_curr(grad_img)

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
            ph_min = np.min(img_cropped.amPh.ph)
            ph_fix = -ph_min if ph_min < 0 else 0
            img_cropped.amPh.ph += ph_fix
            int_matrix = np.copy(img_cropped.amPh.ph)
        else:
            cos_ph_min = np.min(img_cropped.cos_phase)
            cos_ph_fix = -cos_ph_min if cos_ph_min < 0 else 0
            img_cropped.cos_phase += cos_ph_fix
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
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_in_plane = (const.dirac_const / sample_thickness) * (d_phase / d_dist)
        print('{0:.1f} nm'.format(d_dist))
        print('{0:.2f} rad'.format(d_phase))
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
        img = imsup.ImageExp(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'],
                             num=imgNum, px_dim_sz=pxDims[0])
        # img.LoadAmpData(np.sqrt(imgData).astype(np.float32))
        img.LoadAmpData(imgData.astype(np.float32))
        # img.amPh.ph = np.copy(img.amPh.am)
        # ---
        # imsup.RemovePixelArtifacts(img, const.min_px_threshold, const.max_px_threshold)
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
    orig_width = img.width
    crop_width = np.abs(coords[2] - coords[0])
    zoom_factor = orig_width / crop_width
    zoom_img = tr.RescaleImageSki(crop_img, zoom_factor)
    # zoom_img.px_dim *= zoom_factor
    # self.insert_img_after_curr(zoom_img)
    return zoom_img

# --------------------------------------------------------

def modify_image(img, mod=list([0, 0]), is_shift=True):
    if is_shift:
        mod_img = imsup.shift_am_ph_image(img, mod)
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