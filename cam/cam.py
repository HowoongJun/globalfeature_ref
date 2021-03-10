from gcore.hal import *
import common.Log as log
import globalfeature_ref.cam.nets.cam_functions as cam
import globalfeature_ref.cam.nets.resnet as resnet
from globalfeature_ref.cam.nets.utils import preprocess_images
from globalfeature_ref.cam.nets.pooling_functions import weighted_cam_pooling, sum_pooling
import torch
import numpy as np
from PIL import Image

class CModel(CVisualLocalizationCore):
    def __init__(self):
        self.__gpuFlag = False
        self.__batch_size = 50

    def __del__(self):
        log.DebugPrint().info("CAM Destructor")

    def Open(self):
        log.DebugPrint().info("CAM Open")

    def Close(self):
        log.DebugPrint().info("CAM Close")
    
    def Read(self):
        model = resnet.resnet50(pretrained=True)
        model = torch.nn.DataParallel(model)
        data = preprocess_images(self.__Image, 1024, 720, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        features, cams, _, _ = cam.extract_feat_cam(model, "ResNet50", self.__batch_size, data, 64)
        result = weighted_cam_pooling(features, cams)
        return result

    def Write(self, strDbPath, strImgName):
        log.DebugPrint().info("CAM Write")

    def Setting(self, eCommand:int, Value = None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = Image.fromarray(np.array(Value))

        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)

        elif(SetCmd == eSettingCmd.eSettingCmd_CONFIG):
            self.__gpuFlag = Value

    def Reset(self):
        log.DebugPrint().info("CAM Reset")
