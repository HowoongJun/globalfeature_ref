from gcore.hal import *
import numpy as np
from PIL import Image
import globalfeature_ref.gem.nets as gem
import common.Log as log
from globalfeature_ref.gem.utils.general import eCallCmd
import globalfeature_ref.gem.utils.general as general
import os

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
        result = None
        if(self.__gemMode == eCallCmd.eCall_gemPoolFC):
            result = gem.gemPoolFC(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag)
        elif(self.__gemMode == eCallCmd.eCall_gemPoolLw):
            result = gem.gemPoolLw(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag)
        elif(self.__gemMode == eCallCmd.eCall_MacPoolImgNet):
            result = gem.macPoolImgNet(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag)
        return result

    def Write(self, strDbPath, strImgName):
        strDescPath = strDbPath + "/gem"
        if(self.__gemMode == eCallCmd.eCall_gemPoolFC):
            strDescPath = strDescPath + "/GemPoolFC/"
            general.checkFolder(strDescPath)
            np.save(strDescPath + "/" + strImgName, gem.gemPoolFC(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag))

        elif(self.__gemMode == eCallCmd.eCall_gemPoolLw):
            strDescPath = strDescPath + "/GemPoolLw/"
            general.checkFolder(strDescPath) is True
            np.save(strDescPath + "/" + strImgName, gem.gemPoolLw(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag))
            
        elif(self.__gemMode == eCallCmd.eCall_MacPoolImgNet):
            strDescPath = strDescPath + "/MacPoolImgNet/"
            general.checkFolder(strDescPath) is True
            np.save(strDescPath + "/" + strImgName, gem.macPoolImgNet(self.__Image, 512, [1, 1/np.sqrt(2), 1/2], self.__gpuFlag))
            
        return True

    def Setting(self, eCommand:int, Value = None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = Image.fromarray(np.array(Value))
        
        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)
        
        elif(SetCmd == eSettingCmd.eSettingCmd_CONFIG):
            self.__gpuFlag = Value

        elif(SetCmd == eSettingCmd.eSettingCmd_GEM):
            self.__gemMode = Value

    def Reset(self):
        log.DebugPrint().info("GeM Reset!")
