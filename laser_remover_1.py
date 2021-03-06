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


def get_params(y): 
    A=(np.max(y)-np.min(y))/2
    B=(np.max(y)+np.min(y))/2
    freqs = np.fft.fftfreq(len(y))
    index=np.where(rfft(y) == np.min(rfft(y)))[0]
    n=1/(2*np.pi*freqs[index])[0]
    #print(A,B,n)
    return A,B,n
     

def plot_chi_prog(y,n,p0):
    x=np.arange(0+n,y.shape[0]+n,1)
    popt,pcov=curve_fit(sin1,x,y,p0=p0)
    return r2_score(y,sin1(x,*popt)),y-sin1(x,*popt),popt

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
    #y=frame.iloc[:100,10].values   
    popt=get_params(y)   
    print(popt)
    chi,pop=plot_chi_prog(y,popt)
    return popt

def plot_chi_prog(y,popt):
    x=np.arange(0,y.shape[0],1)
    A,fi,n,B=popt
    popt1=-A,fi,n,B
    try:
        popt,pcov=curve_fit(sin1,x,y,p0=popt,
                        bounds=([0.8*A,0.2*fi,0.5*n,0.9*B],[2.2*A,1.8*fi,1.5*n,1.2*B]))
        print('correct 0',popt)
    except:   
        popt1,pcov=curve_fit(sin1,x,y,p0=popt1,
                         bounds=([-0.8*A,0.2*fi,0.5*n,0.9*B],[-2.2*A,1.8*fi,1.5*n,1.2*B]))
        print('correct 1',popt1)
    
    if chisquare(y,sin1(x,*popt))<chisquare(y,sin1(x,*popt1)):
        return chisquare(y,sin1(x,*popt)),popt
    else:
        return chisquare(y,sin1(x,*popt1)),popt1


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
X=np.arange(0,15797.76,15797.76/int(v[-1]))
plt.plot(np.log(frame.iloc[476,:].values))

length,length_g=frame.shape[0],frame_g.shape[0]
w,w_g= signal.blackmanharris(length),signal.blackmanharris(length_g)
spectra=frame.apply(lambda y: make_spectrum(y,w))

plt.plot(X[p1:p2],np.log(spectra.iloc[p1:p2,20:100].T.sum().iloc[:1000].values))

spectra_g=frame_g.apply(lambda y: make_spectrum(y,w_g))
X_g=np.arange(0,15797.76,(15797.76/(spectra_g.shape[0]))).astype(int)

plt.plot(np.log(spectra.iloc[500,:].values))
plt.plot(np.log(spectra.iloc[502,:].values))
plt.plot(np.log(spectra.iloc[504,:].values))

plt.plot(np.log(spectra.iloc[746,:].values))

#X=np.arange(0,15797.76,(15797.76/(spectra.shape[0]))).astype(int)
p1=10
p2=1000


plt.plot(X[p1:p2],np.log(spectra.iloc[p1:p2,1].values))
plt.plot(X[p1:p2],np.log(spectra.iloc[p1:p2,10].values))
plt.plot(X[p1:p2],np.log(spectra.iloc[p1:p2,100].values))
plt.plot(X[p1:p2],np.log(spectra.iloc[p1:p2,200].values))
plt.xlim(0,6000)
plt.plot(X_g[10:1000],np.log(spectra_g.iloc[10:1000,1].values))
plt.xlim(0,6000)
sns.heatmap(np.log(spectra.iloc[100:800,:].values),cmap='hsv')
sns.heatmap(np.log(spectra.iloc[100:800,:].values),cmap='tab20b')
sns.heatmap(np.log(spectra_g.iloc[100:800,:].values),cmap='hsv')
sns.heatmap(np.log(spectra_g.iloc[100:800,:].values),cmap='tab20b')
plt.plot(np.log(spectra.iloc[p1:p2,1].values))
plt.plot(np.log(spectra.iloc[400:800,1].values))
plt.plot(np.log(spectra.iloc[460,:].values))
plt.plot(np.log(spectra.iloc[610,:].values))

frame1=model1(frame)

y=frame.iloc[:500,1].values
x=np.arange(0,y.shape[0],1)
popt[2]=0.6

plt.plot(y)
plt.xlim(0,500)
plt.plot(sin1(x,*popt))
popt[1]=7
plt.plot(y-sin1(x,*popt))
plt.xlim(0,1000)

p1=0
p2=40
popt=get_params(y)
y=frame.iloc[p1:p2,1].values
popt=optimization(y)
x=np.arange(0,y.shape[0],1)
plt.plot(y)
plt.xlim(p1,p2)
plt.plot(sin1(x,*popt))

y=frame.iloc[:500,1].values
x=np.arange(0,y.shape[0])
x1=A*np.sin(n*x+fi)+B
x2=A*np.sin(n*x+fi+1)+B
x3=A*np.sin(n*x+fi+3)+B
x4=A*np.sin(n*x+fi+5)+B
x5=A*np.sin(n*x+fi+7)+B
x6=A*np.sin(n*x+fi+9)+B
z=np.vstack((x1,x2,x3,x4,x5,x6))
z=normalize(z)
z=np.transpose(z)
y=y/(np.max(y)-np.min(y))

svr = SVR() 
svr.fit(z, y)
plt.plot(y)
plt.plot(svr.predict(z))
plt.xlim(800,1200) 

print((chisquare(y,svr.predict(z)))[0])

rf=RandomForestRegressor(n_estimators=10)
rf.fit(z, y)
plt.plot(y)
plt.plot(rf.predict(z))
y=frame.iloc[:,1].values
x=np.arange(0,y.shape[0])
x1=A*np.sin(n*x+fi)+B
x2=A*np.sin(n*x+fi+1)+B
x3=A*np.sin(n*x+fi+3)+B
x4=A*np.sin(n*x+fi+5)+B
x5=A*np.sin(n*x+fi+7)+B
x6=A*np.sin(n*x+fi+9)+B
z=np.vstack((x1,x2,x3,x4,x5,x6))
z=normalize(z)
z=np.transpose(z)
y=y/(np.max(y)-np.min(y))
plt.xlim(800,1200) 
plt.plot(y-rf.predict(z))
