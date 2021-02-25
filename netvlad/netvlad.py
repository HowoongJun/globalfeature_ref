from gcore.hal import *
from gcore.ImageTopic import CImageTopic
from gcore.ImageTopic import ePixelFormat
import globalfeature_ref.netvlad.nets as nets

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
        NetVLADOutput = self.sess.run(self.net_out, feed_dict={self.image_batch: self.__Image.Data.astype('uint8')})
        readOutData.NetVLADVector = NetVLADOutput.astype('float32')

    def Write(self):
        print("Netvlad Write!")

    def Control(self, bGPUFlag = False, oImage):
        tf.reset_default_graph()       
        self.__Image = oImage
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

        if(self.__Image.PixelFormat == ePixelFormat.eRGB8):
            self.image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        elif(self.__Image.PixelFormat == ePixelFormat.eGRAY8):
            self.image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        
        self.net_out = nets.vgg16NetvladPca(self.image_batch)

        saver = tf.train.Saver()
        saver.restore(self.sess, nets.defaultCheckpoint())

    def Reset(self):
        print("Netvlad Reset!")
