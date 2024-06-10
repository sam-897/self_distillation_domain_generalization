import librosa
import numpy as np
import pandas as pd
import pkg_resources
import os
def feats_mel(audio_file):
    num_frames=128
    y, sr = librosa.load(audio_file,sr=22050, mono=True)
    y_norm = librosa.util.normalize(y)
    mels = librosa.feature.melspectrogram(y=y_norm, sr=sr)
    mel_db=librosa.power_to_db(mels)
    return mel_db
labels=pd.read_csv(r"path\PA\PA\ASVspoof2019_PA_cm_protocols\ASVspoof2019.PA.cm.dev.trl.txt",sep=' ',header=None)
y_temp1=labels.iloc[:,[1,4]]
y_temp1.set_index(1, inplace=True)
import numpy as np
folder_path1 = r"path\PA\PA\ASVspoof2019_PA_dev\flac"
Xr=np.zeros((128,1))
Xs=np.zeros((128,1))
ys=[]
yr=[]
fi=0
ri=0
nas=0
for filename in y_temp1.index:
            print(nas)
            if fi//128>=5000 and ri//128>=5000:
              break
            audio_file = os.path.join(folder_path1, filename+".flac")
            mel_db = feats_mel(audio_file)
            label=y_temp1.loc[filename][4]
            if label=="spoof":
                if fi//128>=5000:
                  continue
                Xs=np.hstack((Xs,mel_db))
                fi+=mel_db.shape[1]
            if label=="bonafide":
                if ri//128>=5000:
                  continue
                Xr=np.hstack((Xr,mel_db))
                ri+=mel_db.shape[1]
            nas+=1
print("done extracting")
Xs=np.delete(Xs,0,axis=1)
num_chunkss = Xs.shape[1] // 128
print("spoof chunks",num_chunkss)
Xr=np.delete(Xr,0,axis=1)
num_chunksr = Xr.shape[1] // 128
print("real chunks",num_chunksr)
split_arrays=[]
if num_chunksr>5000:
    c_r=5000
else:
    c_r=num_chunksr
if num_chunkss>5000:
    c_s=5000
else:
    c_s=num_chunkss
for i in range(c_s):
    start_index = i * 128
    end_index = (i + 1) * 128
    chunk = Xs[:, start_index:end_index]
    split_arrays.append(chunk)
spoof=np.array(split_arrays)
print("spoof cut up")
print(spoof.shape)
split_arrays=[]
for i in range(c_r):
    start_index = i * 128
    end_index = (i + 1) * 128
    chunk = Xr[:, start_index:end_index]
    split_arrays.append(chunk)
real=np.array(split_arrays)
print("real cut up")
print(real.shape)
x=np.concatenate((real,spoof))
np.save(r"path\PA_MEL_STACK_X_V2",x)
y1=[]
y2=[]
for i in range(c_s):
    y2.append(1)
for i in range(c_r):
    y1.append(0)
ypa=np.concatenate((y1,y2))
print(ypa.shape)
np.save(r"path\PA_MEL_STACK_Y_V2",ypa)

