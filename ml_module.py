import sys, getopt
import os,json
import pandas as pd
import numpy as np
#import tensorflow as tf

class ML_Module:
    def __init__(self, p1, p2):
        self.hyperparameter={'p1':p1, 'p2':p2}
    
    def load_model(self):
        print ("loading model")
        payload={'model':{'status':'loaded'}}
        return payload
    def predict(self, ifile):
        print ('predicting...')
        print (f'ifile: {ifile}')
        payload={'prediction':0.9}
        return payload
    def train(self, ifile):
        print ('training...')
        print (f'ifile: {ifile}')
        payload={'trained':{'acc':0.9, 'loss':0.001}}
        return payload


def main(argv):
   inputfile = ''

   opts, args = getopt.getopt(argv,"m:i:",["mode=","ifile="])
   for opt, arg in opts:
      if opt in ("-i", "--ifile"):
         ifile = arg
      if opt in ("-m", "--mode"):
         mode = arg
   print (f'Mode:{mode}, ifile:{ifile}')
   if mode=='train':
         ml=ML_Module(p1=1,p2=2)
         ml.train(ifile)
   elif mode=='predict':
         ml=ML_Module(p1=1,p2=2)
         ml.load_model()
         ml.predict(ifile)


if __name__ == "__main__":
   main(sys.argv[1:])