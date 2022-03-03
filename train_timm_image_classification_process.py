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
import torch.cuda
from ikomia import core, dataprocess
from ikomia.dnn import datasetio, dnntrain
from ikomia.core.task import TaskParam
import copy
# Your imports below
from train_timm_image_classification.utils import train, parser
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainTimmImageClassificationParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.cfg["model_name"] = "resnet18"
        self.cfg["custom_cfg"] = False
        self.cfg["cfg_path"] = ""
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 4
        self.cfg["input_size"] = [224, 224]
        self.cfg["pretrained"] = True
        self.cfg["output_folder"] = ""
        self.cfg["learning_rate"] = 0.05

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["custom_cfg"] = eval(param_map["custom_cfg"])
        self.cfg["cfg_path"] = param_map["cfg_path"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["input_size"] = eval(param_map["input_size"])
        self.cfg["pretrained"] = eval(param_map["pretrained"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["learning_rate"] = float(param_map["learning_rate"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["model_name"] = self.cfg["model_name"]
        param_map["custom_cfg"] = str(self.cfg["custom_cfg"])
        param_map["cfg_path"] = self.cfg["cfg_path"]
        param_map["epochs"] = str(self.cfg["epochs"])
        param_map["batch_size"] = str(self.cfg["batch_size"])
        param_map["input_size"] = str(self.cfg["input_size"])
        param_map["pretrained"] = str(self.cfg["pretrained"])
        param_map["output_folder"] = self.cfg["output_folder"]
        param_map["learning_rate"] = self.cfg["learning_rate"]
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainTimmImageClassification(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # remove default input
        self.removeInput(0)
        self.addInput(dataprocess.CPathIO(core.IODataType.FOLDER_PATH))
        self.stop_train = False
        self.tb_writer = None
        # Create parameters class
        if param is None:
            self.setParam(TrainTimmImageClassificationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        self.stop_train = False
        data_dir = self.getInput(0).getPath()

        if not (os.path.isdir(data_dir)):
            print("Input for train_timm_image_classification plugin is not correct. "
                  "Make sure to put a valid path as input")

        if not param.cfg["custom_cfg"]:
            args = parser.parse_args([data_dir])
            args.output = os.path.dirname(__file__) + "/output" if param.cfg["output_folder"] == "" else param.cfg[
                "output_folder"]
            if not os.path.isdir(args.output):
                os.makedirs(args.output)

            class_names = []
            for base, dirs, files in os.walk(data_dir + "/train"):
                for directory in dirs:
                    class_names.append(directory)
            args.model = param.cfg["model_name"]
            args.pretrained = param.cfg["pretrained"]
            args.num_classes = len(class_names)
            args.batch_size = param.cfg["batch_size"]
            args.epochs = param.cfg["epochs"]
            args.prefetcher = not args.no_prefetcher
            args.input_size = [3] + param.cfg["input_size"]
            args.lr = param.cfg["learning_rate"]
            args.checkpoint_hist = 5
            args.cooldown_epochs = 10 if args.epochs > 10 else 0
            args.epochs = args.epochs - args.cooldown_epochs
            args.no_aug = True

        elif os.path.isfile(param.cfg["cfg_path"]):
            with open(param.cfg["cfg_path"], 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
                args = parser.parse_args([data_dir])
            class_names = []
            for base, dirs, files in os.walk(args.data_dir + "/train"):
                for directory in dirs:
                    class_names.append(directory)
        else:
            print("Config file " + param.cfg["cfg_file"] + " does not exist")

        # Distributed training not yet available in Ikomia
        args.distributed = False
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        args.world_size = 1
        args.rank = 0  # global rank
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        tb_logdir = self.getTensorboardLogDir() + "/" + param.cfg["model_name"] + str(param.cfg["input_size"][0]) \
                    + "/" + str_datetime
        self.tb_writer = SummaryWriter(tb_logdir)
        train(args, self.get_stop, self.tb_writer, self.log_metrics, class_names)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainTimmImageClassificationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_timm_image_classification"
        self.info.shortDescription = "Train timm image classification models"
        self.info.description = "Train timm image classification models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Train"
        self.info.iconPath = "icons/timm.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Ross Wightman"
        self.info.article = "PyTorch Image Models"
        self.info.journal = "GitHub repository"
        self.info.year = 2019
        self.info.license = "Apache-2.0 License "
        # URL of documentation
        self.info.documentationLink = "https://rwightman.github.io/pytorch-image-models/"
        # Code source repository
        self.info.repository = "https://github.com/rwightman/pytorch-image-models"
        # Keywords used for search
        self.info.keywords = "image, classification, imagenet, pretrain, pytorch"

    def create(self, param=None):
        # Create process object
        return TrainTimmImageClassification(self.info.name, param)
