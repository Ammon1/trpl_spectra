import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit
from scipy import fftpack
from scipy import signal
from scipy import interpolate
from scipy.fftpack import fft
from scipy.signal import blackman

    
def fft_frame(df):
    x=df.iloc[:,1].values
    y=np.abs(fft(x))
    length=y.shape[0]
    w = blackman(length)
    spectra=df.apply(lambda x: np.abs(fft(w*x)))
    return spectra
    
    
def read_files(path,filename):
    
    frame=pd.read_csv(path+'\\'+filename,index_col=False,sep=' ')
    cols=np.arange(0,frame.shape[1])
    frame=pd.read_csv(path+'\\'+filename,index_col=False,sep=' ',names=cols)
    return frame

def model(x,x_interp,y,interpolate):
    model=interpolate.interp1d(x, y,kind='cubic')
    return model(x_interp)

def interp(df,interpolation):
    x=np.arange(0,df.shape[0])
    x_interp=np.arange(0,x[-1],interpolation)
    df_interp=df.apply(lambda y: model(x,x_interp,y,interpolate))
    return df_interp
    
def interpolate_df(df,interpolation):
    x=np.arange(0,df.shape[0])
    x_interp=np.arange(0,x[-1],interpolation)
    df_int=pd.DataFrame(np.zeros((x_interp.shape[0],df.shape[1])))
    for col in df.columns:
            y=df.loc[:,col].values
            model=interpolate.interp1d(x, y,kind='cubic')
            df_int.loc[:,col]=model(x_interp)
           
    return df_int  

def make_spectrum(frame):
    spectrum=fft_frame(frame)
    #spectrum=spectrum.T
    x=np.arange(0,2*15797.76,(2*15797.76/(spectrum.shape[0]))).astype(int)
    y=np.arange(0,(25/1000)*spectrum.shape[1],(25/1000))
    x=np.around(x,decimals=2)
    y=np.around(y,decimals=2)
    spectrum.columns=y
    spectrum=spectrum.set_index(x)
    spectrum=spectrum.T
    #fig = plt.figure(figsize=(6,6), dpi=200)
    sns.heatmap(np.log(spectrum.iloc[:,100:300]),cmap='hsv')
    return spectrum,x,y

from scipy.signal import savgol_filter   
def func(x,x0,A,tau):
    return A*np.exp(-(x-x0)/tau)

path =r'C:\Users\Administrator\Desktop\Kacper\2019\11\5'
filename='HgCdTe_70K'
frame=read_files(path,filename)
frame=frame.groupby(0, as_index=False).mean()
frame_int=interp(frame,1/interp)
sns.heatmap(frame.iloc[:,100:500],cmap='hsv') 


spectrum,x,y=make_spectrum(frame)
plt.plot(np.log(spectrum.loc[:,2186]))

x_plot=y[1000:4000]
y_plot= spectrum.loc[25:99.98,2186].values

popt,pcov=curve_fit(func, x_plot, y_plot,p0=[0,40000,20]
                    ,bounds=([0,0,0],[25,1e6,200]))
plt.plot(x_plot,y_plot)
plt.plot(x_plot,func(x_plot,*popt))


