from gcore.hal import *
import numpy as np
from PIL import Image
import globalfeature_ref.gem.nets as gem
import common.Log as log

class CModel(CVisualLocalizationCore):
    def __init__(self):
        self.__gpuFlag = False

    def __del__(self):
        log.DebugPrint().info("GeM Destructor!")

    def Open(self):
        log.DebugPrint().info("GeM Open")
    
    def Close(self):
        log.DebugPrint().info("GeM Close!")

    def Read(self):
        log.DebugPrint().info("GeM Read!")
        gem.gemPoolFC(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag)
        gem.gemPoolLw(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag)
        gem.macPoolImgNet(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag)

    def Write(self):
        log.DebugPrint().info("GeM Write!")

    def Setting(self, eCommand:int, Value = None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = Image.fromarray(np.array(Value))
        
        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)
        
        elif(SetCmd == eSettingCmd.eSettingCmd_CONFIG):
            self.__gpuFlag = Value

    def Reset(self):
        log.DebugPrint().info("GeM Reset!")
