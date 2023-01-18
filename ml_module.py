import sys, getopt
import os,json
import pandas as pd
import numpy as np
#from deepface import DeepFace 

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


def recognize (path_to_imagefile):
    with open('data_deepface.json', 'r') as f:
        dic_pre_computed_embedings=json.load(f)

    embeddings = DeepFace.represent(img_path = path_to_imagefile, model_name = 'Facenet512',enforce_detection=False)
    df_score = pd.DataFrame(columns=['name','score'])
    for name in dic_pre_computed_embedings:
        aa=dic_pre_computed_embedings[name]
        for inst in range (len(aa)):
            pre_computed=aa[inst][1]
            score=DeepFace.dst.findCosineDistance(embeddings, pre_computed)
            df_score_row = pd.DataFrame([[name,score]], columns=['name','score'])
            df_score=pd.concat([ df_score,df_score_row])
            df_final=df_score.sort_values('score').head(3)
    #print (df_final) 
    #payload=df_final.reset_index().to_json()
    response_json=df_final.to_dict('list')
    return response_json


def main(argv):
   inputfile = ''
   outputfile = ''
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
         #print (recognize(ifile))


if __name__ == "__main__":
   main(sys.argv[1:])