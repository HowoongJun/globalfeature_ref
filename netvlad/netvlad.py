from gcore.hal import *
import globalfeature_ref.netvlad.nets as nets
import numpy as np
import tensorflow.compat.v1 as tf

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("Netvlad constructor called")

    def __del__(self):
        print("Netvlad Destructor!")

    def Open(self):
        print("Netvlad Open")
    
    def Close(self):
        print("Netvlad Close!")

    def Read(self):
        NetVLADOutput = self.sess.run(self.net_out, feed_dict={self.__image_batch: self.__Image})
        result = NetVLADOutput.astype('float32')
        return result

    def Write(self):
        print("Netvlad Write!")

    def Control(self, oImage, bGPUFlag = False):
        tf.reset_default_graph()       
        self.__Image = np.expand_dims(oImage, axis = 0)
        height, width, channels = oImage.shape
        self.sess = None
        if bGPUFlag == False:
            print('Using CPUs..')
            config = tf.ConfigProto(device_count = {'GPU': 0})
            self.sess = tf.Session(config=config)
        else:
            print('Using GPUs..')
            self.sess = tf.Session()

        if self.sess is None:
            return False

        if(channels == 3):
            self.__image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        elif(channels == 1):
            self.__image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        
        self.net_out = nets.vgg16NetvladPca(self.__image_batch)

        saver = tf.train.Saver()
        saver.restore(self.sess, nets.defaultCheckpoint())

    def Reset(self):
        print("Netvlad Reset!")
