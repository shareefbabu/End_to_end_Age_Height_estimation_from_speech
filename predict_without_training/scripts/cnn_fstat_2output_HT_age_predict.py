'''This code is to predict the age and height
of the speaker using 60 MFCC features..
You need to pass path of the input feature file.
This will predict the lables corresponding to No. of 
frames and shift which you specified. 
The predicted lables are printed a file.
These models are trained gender specific, change the trained
models and weight files corresponding to the gender files 
which needs to be predicted
Usage: python cnn_fstat_2output_HT_age_predict sre_test_ht_age.list predicted_lables.out 150 75
'''

from __future__ import print_function
import keras
import sys
import os.path
import h5py
import time as time
from keras.datasets import mnist
from keras.models import Model,Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Input,regularizers 
from keras.layers import Conv1D, MaxPooling1D, Lambda ,merge
from keras import backend as K
import htkmfc as htk
import numpy as np
from keras.layers.normalization import BatchNormalization
t = time.time()

testfile_path = str(sys.argv[1]) 
predicted_labels  = str(sys.argv[2])
Nframes = int(sys.argv[3]) 
NfrShift = int(sys.argv[4])

#Nframes = 150
#NfrShift = 75
def mean_time(x):
    x = K.mean(x, axis=1)
    return x 

def tensor_prod(x):
    mfcc = x[0]; #batch, None, 60
    xp = x[1];   #batch, None, 256
    return xp[..., None] * mfcc[..., None, :]



def load_htkfile_full(input_file):
    feat_reader = htk.open(input_file)			#extracting features from the htk files
    feat1  = feat_reader.getall();
    feat = np.reshape(feat1,(1,-1,60))
    return feat
    
def my_init(shape, dtype=None):
    mean_file=str('../model/means')
    mean_gmm = np.loadtxt(mean_file, dtype=np.object, delimiter=" ")#dimension= 256X60
    mean_gmm = mean_gmm.astype(np.float) 
    eq_norm = np.sqrt(np.sum(mean_gmm**2, axis=1))
    norm_mean = np.transpose(mean_gmm/eq_norm[:, np.newaxis])
    #kbias = np.zeros((256,))
    kvar = K.variable(value=norm_mean, dtype='float32')
    
    return kvar


def pred_feat_seg(input_list,Nframes):
    List_inp = open(input_list).readlines()
    trgt=np.empty(shape=[0,2])
    pred_1=np.empty(shape=[0,2])
    man_pred_all=np.empty(shape=[0,2]) 
    speaker=[]
    feat_all= np.empty(shape=[0,0, 60])
    for in_file in range (len(List_inp)) :
        f = (List_inp[in_file]).split()
        spkrid_split  = f[0].split('/')
        spkrid_split1  = spkrid_split[4].split('_')
        spkrid  = spkrid_split1[0]
        print(f[0], spkrid)
        feat=load_htkfile_full(f[0])
        #trgt_ht= f[1]
        #trgt_age= f[2]
        #trgt_ht_age =np.array([[trgt_ht,trgt_age]])
        #trgt=np.append(trgt,trgt_ht_age, axis=0)
        N = feat.shape[1];
        print('shape of feat',N)
        if(N > Nframes):
            x = np.empty(shape = [0,2]);
	    j=0;
            for i in np.arange(0,N-Nframes+1,NfrShift):
                y=statnw.predict(feat[:,i:i+Nframes,:])
                x = np.append(x,y,axis=0)
                j= j+1
            pred_1 = x
             
	    spkrid_N = '{} '.format(spkrid) *j
	    A=spkrid_N.split(' ')[:-1]
	    #pred_1 = np.median(x,axis=0);
            #pred_1 = np.mean(x,axis=0);
        else:
            pred_1 = statnw.predict(feat)
        man_pred = pred_1
        spkrid_N = spkrid  
	#print(spkrid_N)        
	man_pred_all =  np.append(man_pred_all,man_pred)
        man_pred_all = np.reshape(man_pred_all,(-1,2))
        speaker.extend(A)
        
    return speaker, man_pred_all
 
mfcc = Input((None,60)); 
x = Conv1D(256, kernel_size= 1, activation='relu')(mfcc)
x = BatchNormalization()(x)
x = Dropout(0.3)(x) 
x = Conv1D(512, kernel_size= 1, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x) 
x = Conv1D(256, kernel_size= 1, activation='softmax')(x)
xp = Dropout(0.3)(x) 
supvec = Lambda(tensor_prod)([mfcc,xp]) 
fstat1 = Lambda(mean_time)(supvec) 
fstat2 = Flatten()(fstat1)
fstat2 = Dropout(0.3)(fstat2)
pred = Dense(2,activation='linear')(fstat2)

statnw = Model(input=mfcc,output=pred)


statnw.compile(loss='mse',
              optimizer=keras.optimizers.Adam(lr=0.0002),
              metrics=['mse'])

statnw.summary()



saved_nn = Sequential()
saved_nn.add(Dense(256, input_dim=60, activation='relu', kernel_initializer=my_init))
saved_nn.add(BatchNormalization())
saved_nn.add(Dropout(0.3))
saved_nn.add(Dense(512, activation='relu'))
saved_nn.add(BatchNormalization())
saved_nn.add(Dropout(0.3))
saved_nn.add(Dense(256,activation='softmax'))
saved_nn.load_weights('../model/part1_trained_model.h5')

#saved_nn.summary()

for i in [0,3,6]:
    gparms = saved_nn.layers[i].get_weights();
    gwts  = gparms[0];
    print("layer", i, gwts.shape)
    gbias = gparms[1];
    swts  = np.zeros((1,gwts.shape[0],gwts.shape[1]))
    swts[0,:,:] = gwts; 
    statnw.layers[i+1].set_weights([swts,gbias]);
    
for i in [1,4]:
    gparms = saved_nn.layers[i].get_weights();
    statnw.layers[i+1].set_weights(gparms);

svr_file=str('../model/ht_age_svr15360_male.csv')
weights_svr = np.loadtxt(svr_file, dtype=np.object, delimiter=" ")
weights_svr = weights_svr.astype(np.float) 
weights_svr = np.transpose(weights_svr)
print(weights_svr.shape)
gbias_13     = np.loadtxt('../model/ht_age_biasSVR_male.txt')  
gbias_13  = np.array([gbias_13])
gbias_13 = np.reshape(gbias_13,(2,))*-1
swts_13 = weights_svr;
print('BIAS',gbias_13)
gparms_13 = statnw.layers[13].get_weights();
statnw.layers[13].set_weights([swts_13,gbias_13])


#test_inpFile = str('../list/sre_test_ht_age.list')
    
model_name_weights = "male_HT_AGE_part3_model.h5"
print(model_name_weights)
model_path =  '../model/' + model_name_weights
print(model_path)
statnw.load_weights(model_path)
print('Number of frames',Nframes)   
spkrID,ts_pred_all = pred_feat_seg(testfile_path,Nframes)        
ts_pred_all = ts_pred_all.astype(np.float)
#ts_trgt_all = ts_trgt_all.astype(np.float)
spkrID_np=np.array(spkrID)
predictions=ts_pred_all.tolist()

outputlist=[]
outfile=open(predicted_labels,'a+')
for speaker,eachpred in zip(spkrID,predictions):
    #print("$$ ",speaker,eachpred)
    outputlist.append('{} {} {}\n'.format(speaker,eachpred[0],eachpred[1]))

outfile.writelines(outputlist)

elapsed = time.time() - t
print ('time taken :', elapsed)
