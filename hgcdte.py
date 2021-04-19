# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:36:09 2020

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.optimize import curve_fit
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import rfft
from scipy.signal import blackman
from sklearn.svm import SVR
from scipy.stats import chisquare
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

  
def read_files(path,filename):
    
    frame=pd.read_csv(path+'\\'+filename,index_col=False,sep=' ')
    return frame

def make_spectrum(y,w):
    return np.abs(rfft(y.iloc[:].values*w))

def sin1(x,A0,fi0,n0,B):
    return A0*np.sin(x*n0-fi0)+B


def preprocessing(popt,shape):
    x=np.arange(0,shape,1)
    x1=np.full(shape,popt[2])
    x2=sin1(x,*popt)
    z=np.vstack((x,np.power(x,2),np.power(x,3),x-500,x-1000,x-1500,x-2000,-x+500,-x+1500,-x+2000,x1,x2))
    z=normalize(z)
    z=np.transpose(z)
    return z

def fitting(z,y):
    epsilon=0.002
    C=1000
    svr = SVR(epsilon=epsilon) 
    svr.fit(z, y) 
    spectrum=y-svr.predict(z)
    print((chisquare(y,svr.predict(z)))[0])
    return spectrum,svr

def correction(z,y,svr):
    svr.fit(z, y) 
    spectrum=y-svr.predict(z)
    print('chi ',int((chisquare(y,svr.predict(z)))[0]))
    length=spectrum.shape[0]
    #w= signal.blackmanharris(length)
    #x=np.arange(0,15797.76,(15797.76/(spectrum.shape[0]))).astype(int) 
    return spectrum

def optimization(y):
    popt=get_params(y)   
    popt=plot_chi_prog(y,popt)
    x=np.arange(0,y.shape[0],1)
    print(chisquare(y,sin1(x,*popt))[0])
    return popt

def plot_chi_prog(y,popt):
    x=np.arange(0,y.shape[0],1)
    A,fi,n,B=popt
    popt1=A,fi,n,B
    print('popt ',[round(popt1,2) for popt1 in popt1])
    try:
        popt,pcov=curve_fit(sin1,x,y,p0=popt,
                        bounds=([0.5*A,0.2*fi,0.5*n,-0.01],[1.5*A,np.pi,2*n,0.01]))
        print('corr ',[round(popt,2) for popt in popt])
    except Exception as e:   
        popt,pcov=curve_fit(sin1,x,y,p0=popt,
                         bounds=([-0.8*A,0,0.5*n,-0.1],[-1.5*A,2*3.14,1.5*n,0.1]))    
                        # bounds=([-0.8*A,0.2*fi,0.5*n,-0.1],[-2.2*A,1.8*fi,1.5*n,0.1]))
        print('corr1',[round(popt1,2) for popt in popt])
    
    plt.plot(y)
    plt.plot(sin1(x,*popt))
    return popt
    
    #if chisquare(y,sin1(x,*popt))<chisquare(y,sin1(x,*popt1)):
    #    print('0')
    #    plt.plot(y)
    #    plt.plot(sin1(x,*popt))
    #    return chisquare(y,sin1(x,*popt)),popt
    #else:
    #    print('1')
    #    return chisquare(y,sin1(x,*popt1)),popt1


def get_params(y): 
    A=(np.max(y)-np.min(y))/2
    B=np.mean(y)
    freqs = np.fft.fftfreq(len(y))
    index=np.where(rfft(y) == np.max(rfft(y)[1:]))[0]
    n=np.pi*freqs[index][0]
    
    rest0=1e8
    fi0=0
    for fi in range(0,60,1):
        fi=fi/10
        rest=np.abs(y[0]-(A*np.sin(fi*n)+B))
        if rest<rest0:
            rest0=rest
            fi0=fi
    popt=[A,fi0,n,B]
    return popt

def model1(frame):    
    #scalling
    y=frame.iloc[:,1]
    #finding z array
    popt=optimization(y)
    z=preprocessing(popt,frame.shape[0])
    #train svr model
    spectrum,svr=fitting(z,y)
    frame1=frame.apply(lambda y: correction(z,y,svr))
    return frame1


#######algorhitm
path =r'C:\Users\Administrator\Desktop\Kacper\2020\9\15'
filename='\\test_hgcdte1'
frame=read_files(path,filename)
frame_g=frame.groupby(frame.columns.values[0], as_index=False).mean()
frame.isna().sum().sum()
v=frame[frame.columns.values[0]].values

length,length_g=frame.shape[0],frame_g.shape[0]
w,w_g= signal.blackmanharris(length),signal.blackmanharris(length_g)
spectra=frame.apply(lambda y: make_spectrum(y,w))
spectra_g=frame_g.apply(lambda y: make_spectrum(y,w_g))
X_g=np.arange(0,15797.76,(15797.76/(spectra_g.shape[0]))).astype(int)

X=np.arange(1,15797.76,15797.76/spectra.shape[0])
X=np.around(X,decimals=2)
spectra.columns=np.arange(0,spectra.shape[1])
spectra=spectra.set_index(X)

plt.plot(X[p1:p2],np.log(spectra.iloc[p1:p2,20:100].T.sum().iloc[:1000].values))
plt.plot(X_g[p1:p2],np.log(spectra_g.iloc[p1:p2,20:100].T.sum().iloc[:1000].values))


plt.plot(np.log(spectra.iloc[p1:p2,40].values))
plt.plot(np.log(spectra.iloc[510,40:]))
#X=np.arange(0,15797.76,(15797.76/(spectra.shape[0]))).astype(int)
p1=10
p2=1000

plt.plot(np.log(spectra.iloc[p1:p2,40]))
plt.plot(np.log(spectra.iloc[p1:p2,100]))
plt.plot(np.log(spectra.iloc[p1:p2,350]))
plt.xlim(0,5000)
plt.plot(X_g[10:1000],np.log(spectra_g.iloc[10:1000,1].values))
plt.xlim(0,6000)
sns.heatmap(np.log(spectra.iloc[10:1000,:]),cmap='hsv')
sns.heatmap(np.log(spectra.iloc[100:1000,:]),cmap='tab20b')
sns.heatmap(np.log(spectra_g.iloc[100:1000,:].values),cmap='hsv')
sns.heatmap(np.log(spectra_g.iloc[100:1000,:].values),cmap='tab20b')

plt.plot(frame.iloc[1:,40].values)
freqs = np.fft.fftfreq(len(frame.iloc[1:800,40].values))
plt.plot(freqs*np.pi,np.abs(rfft(frame.iloc[1:800,40].values)))
plt.ylim(0,100)
plt.xlim(0.4,0.7)

freqs = np.fft.fftfreq(len(frame.iloc[1300:,40].values))
plt.plot(freqs*np.pi,np.abs(rfft(frame.iloc[1300:,40].values)))
plt.ylim(0,100)
plt.xlim(0.4,0.7)

plt.plot(frame.rolling(20).mean().iloc[1:500,40].values)
plt.plot(savgol_filter(frame.iloc[1:,40].values,101,3)[1:500])

y=frame.iloc[1:,40].values
x=np.arange(0,frame.iloc[1:,40].shape[0])
z=np.polyfit(x,y, 30)
p=np.poly1d(z)
plt.plot(p(x))
plt.plot(frame.iloc[1:,40].values,'.')

y_corr=y
y_corr=frame.iloc[1:,40].values/p(x)-1
popt=get_params(y_corr)
plt.plot(y_corr[700:1400])
plt.plot(sin1(x[:800],*popt))
popt[2]=0.615
popt2=popt
popt2[2]=0.47


popt1=get_params(y_corr[0:100])
popt2=get_params(y_corr[100:200])
popt3=get_params(y_corr[200:300])
popt4=get_params(y_corr[300:400])
popt5=get_params(y_corr[400:500])
popt6=get_params(y_corr[500:600])
popt7=get_params(y_corr[600:700])
popt8=get_params(y_corr[700:800])
popt9=get_params(y_corr[800:900])

popt10=get_params(y_corr[900:1000])

popt17=get_params(y_corr[2000:2100])
popt18=get_params(y_corr[2100:2200])
popt19=get_params(y_corr[2200:2300])
popt20=get_params(y_corr[2300:2400])
popt21=get_params(y_corr[2400:2500])
popt22=get_params(y_corr[2500:2600])
popt23=get_params(y_corr[2600:2700])
popt24=get_params(y_corr[2700:2800])
popt25=get_params(y_corr[2800:2900])
popt26=get_params(y_corr[2900:3000])


popt28=get_params(y_corr[0:200])
popt29=get_params(y_corr[200:400])
popt30=get_params(y_corr[400:600])
popt31=get_params(y_corr[600:800])
popt32=get_params(y_corr[800:1000])

popt38=get_params(y_corr[2000:2200])
popt39=get_params(y_corr[2200:2400])
popt40=get_params(y_corr[2400:2600])
popt41=get_params(y_corr[2600:2800])





sig1=sin1(x,*popt1)
sig2=sin1(x,*popt2)
sig3=sin1(x,*popt3)
sig4=sin1(x,*popt4)
sig5=sin1(x,*popt5)
sig6=sin1(x,*popt6)
sig7=sin1(x,*popt7)
sig8=sin1(x,*popt8)
sig9=sin1(x,*popt9)
sig10=sin1(x,*popt10)
#sig6=sin1(x,*popt11)
#sig7=sin1(x,*popt12)
#sig8=sin1(x,*popt13)
#sig9=sin1(x,*popt14)
#sig10=sin1(x,*popt15)
#sig61=sin1(x,*popt16)
sig11=sin1(x,*popt17)
sig12=sin1(x,*popt18)
sig13=sin1(x,*popt19)
sig14=sin1(x,*popt20)
sig15=sin1(x,*popt21)
sig16=sin1(x,*popt22)
sig117=sin1(x,*popt23)
sig105=sin1(x,*popt24)
sig106=sin1(x,*popt25)
sig107=sin1(x,*popt26)
#sig108=sin1(x,*popt27)
sig109=sin1(x,*popt28)

sig109=sin1(x,*popt29)
sig110=sin1(x,*popt30)
sig111=sin1(x,*popt31)
sig112=sin1(x,*popt32)
#sig113=sin1(x,*popt33)
#sig114=sin1(x,*popt34)
#sig115=sin1(x,*popt35)
#sig116=sin1(x,*popt36)
#sig117=sin1(x,*popt37)
sig118=sin1(x,*popt38)
sig119=sin1(x,*popt39)
sig120=sin1(x,*popt40)
sig121=sin1(x,*popt41)
#sig122=sin1(x,*popt42)
#sig123=sin1(x,*popt43)


z=np.vstack((sig1,sig2,sig3,sig4,sig5, sig11,sig21,sig31,sig41,sig51,
             sig6,sig7,sig8,sig9,sig10,  sig61,sig71,sig81,sig91,sig101,sig102,sig103,
             sig104,sig105,sig106,sig107,sig108,sig109,
             sig110,sig111,sig112,sig113,sig114
             ,sig115,sig116,sig117,sig118,sig119,sig120
             ,sig121,sig122))
z=np.transpose(z)
model=RandomForestRegressor(n_estimators=100,min_samples_split=10,min_samples_leaf=10,n_jobs=-1, 
                            max_features=None,random_state=10)
model.fit(z,y_corr)
model.score(z,y_corr)
plt.plot(y_corr-model.predict(z))
plt.plot(y_corr)
plt.plot(model.predict(z))
plt.xlim(0,1000)
plt.plot(model.feature_importances_)
