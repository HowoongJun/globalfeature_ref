from gcore.hal import * 
import globalfeature_ref.netvlad.nets as nets
import numpy as np
import tensorflow.compat.v1 as tf
import os 
import common.Log as log

class CModel(CVisualLocalizationCore):
    def __init__(self):
        tf.reset_default_graph()

    def __del__(self):
        log.DebugPrint().info("Netvlad Destructor!")

    def Open(self):
        log.DebugPrint().info("Netvlad Open")
    
    def Close(self):
        log.DebugPrint().info("Netvlad Close!")

    def Read(self):
        NetVLADOutput = self.sess.run(self.net_out, feed_dict={self.__image_batch: self.__Image})
        result = NetVLADOutput.astype('float32')
        return result

    def Write(self, strDbPath, strImgName):
        strDescPath = strDbPath + "/netvlad/"
        if(not os.path.exists(strDescPath)):
            os.makedirs(strDescPath)
        np.save(strDescPath + "/" + strImgName, self.Read())
        return True

    def Setting(self, eCommand:int, Value = None): 
        SetCmd = eSettingCmd(eCommand)
        
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = np.expand_dims(np.asarray(Value), axis = 0)
        
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)

        elif(SetCmd == eSettingCmd.eSettingCmd_CONFIG):
            bGPUFlag = Value
            if bGPUFlag == False:
                config = tf.ConfigProto(device_count = {'GPU': 0})
                self.sess = tf.Session(config=config)
            else:
                self.sess = tf.Session()

            if self.sess is None:
                return False

            if(self.__channel == 3):
                self.__image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
            elif(self.__channel == 1):
                self.__image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            
            self.net_out = nets.vgg16NetvladPca(self.__image_batch)

            saver = tf.train.Saver()
            saver.restore(self.sess, nets.defaultCheckpoint())

    def Reset(self):
        tf.reset_default_graph()
