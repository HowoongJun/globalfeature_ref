from gcore.hal import *

#Just for test
import globalfeature_ref.gem.examples.example_descriptor_extraction as gem

class CModel(CVisualLocalizationCore):
    def __init__(self):
        self.__gpuFlag = False

    def __del__(self):
        print("GeM Destructor!")

    def Open(self):
        print("GeM Open")
    
    def Close(self):
        print("GeM Close!")

    def Read(self):
        print("GeM Read!")
        gem.main()

    def Write(self):
        print("GeM Write!")

    def Control(self, oImage, bGPUFlag = False):
        self.__gpuFlag = bGPUFlag

    def Reset(self):
        print("GeM Reset!")
