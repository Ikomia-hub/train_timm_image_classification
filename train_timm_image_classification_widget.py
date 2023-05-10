# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_timm_image_classification.train_timm_image_classification_process import TrainTimmImageClassificationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import timm
from train_timm_image_classification.utils_ui import Autocomplete


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainTimmImageClassificationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainTimmImageClassificationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Model name
        timm_models = timm.list_models()
        self.combo_model = Autocomplete(timm_models, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.gridLayout.addWidget(self.combo_model, 0, 1)
        self.gridLayout.addWidget(self.label_model, 0, 0)

        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])
        # Input size
        self.spin_input_h = pyqtutils.append_spin(self.gridLayout, "Input height", self.parameters.cfg["input_size"][0],
                                                  min=16)
        self.spin_input_w = pyqtutils.append_spin(self.gridLayout, "Input width", self.parameters.cfg["input_size"][1],
                                                  min=16)
        # Epochs
        self.spin_epochs = pyqtutils.append_spin(self.gridLayout, "Epochs", self.parameters.cfg["epochs"], min=1)
        # Batch size
        self.spin_batch_size = pyqtutils.append_spin(self.gridLayout, "Batch size", self.parameters.cfg["batch_size"])
        # Pretrain
        self.check_pretrained = pyqtutils.append_check(self.gridLayout, "Pretrained on Imagenet",
                                                       self.parameters.cfg["use_pretrained"])
        # Backbone
        self.check_backbone = pyqtutils.append_check(self.gridLayout, "Train backbone",
                                                     self.parameters.cfg["train_backbone"])
        # Output folder
        self.browse_output_folder = pyqtutils.append_browse_file(self.gridLayout, "Output folder",
                                                                 self.parameters.cfg["output_folder"],
                                                                 mode=pyqtutils.QFileDialog.Directory)
        # Base learning rate
        self.double_spin_lr = pyqtutils.append_double_spin(self.gridLayout, "Learning rate",
                                                           self.parameters.cfg["learning_rate"], step=1e-4)
        # Custom config
        self.check_custom_cfg = pyqtutils.append_check(self.gridLayout, "Enable expert mode",
                                                       self.parameters.cfg["use_custom_cfg"])

        self.browse_custom_cfg = pyqtutils.append_browse_file(self.gridLayout, "Custom config path",
                                                              self.parameters.cfg["config_file"])
        # Disable unused widgets when custom config checkbox is checked
        self.browse_custom_cfg.setEnabled(self.check_custom_cfg.isChecked())
        self.double_spin_lr.setEnabled(not self.check_custom_cfg.isChecked())
        self.browse_output_folder.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_pretrained.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_batch_size.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_epochs.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_input_w.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_input_h.setEnabled(not self.check_custom_cfg.isChecked())
        self.combo_model.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_backbone.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_custom_cfg.stateChanged.connect(self.on_check)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check(self, int):
        self.browse_custom_cfg.setEnabled(self.check_custom_cfg.isChecked())
        self.double_spin_lr.setEnabled(not self.check_custom_cfg.isChecked())
        self.browse_output_folder.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_pretrained.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_batch_size.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_epochs.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_input_w.setEnabled(not self.check_custom_cfg.isChecked())
        self.spin_input_h.setEnabled(not self.check_custom_cfg.isChecked())
        self.combo_model.setEnabled(not self.check_custom_cfg.isChecked())
        self.check_backbone.setEnabled(not self.check_backbone.isChecked())

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["use_custom_cfg"] = self.check_custom_cfg.isChecked()
        self.parameters.cfg["config_file"] = self.browse_custom_cfg.path
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch_size.value()
        self.parameters.cfg["input_size"] = [self.spin_input_h.value(), self.spin_input_w.value()]
        self.parameters.cfg["use_pretrained"] = self.check_pretrained.isChecked()
        self.parameters.cfg["output_folder"] = self.browse_output_folder.path
        self.parameters.cfg["learning_rate"] = self.double_spin_lr.value()
        self.parameters.cfg["train_backbone"] = self.check_backbone.isChecked()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainTimmImageClassificationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_timm_image_classification"

    def create(self, param):
        # Create widget object
        return TrainTimmImageClassificationWidget(param, None)
