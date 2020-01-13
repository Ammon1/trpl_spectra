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

def ica_plot(X,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5
        y+=X[:,6]*a6+X[:,7]*a7+X[:,8]*a8+X[:,9]*a9
        return y
    
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
    #spectrum.columns=y
    spectrum=spectrum.set_index(x)
    spectrum=spectrum.T
    #fig = plt.figure(figsize=(6,6), dpi=200)
    sns.heatmap(np.log(spectrum.iloc[:,100:300]),cmap='hsv')
    return spectrum,x,y

from scipy.signal import savgol_filter   
def func(x,x0,x1,A0,A1,tau0,tau1):
    return A0*np.exp(-(x-x0)/tau0)+A1*np.exp(-(x-x1)/tau1)


path =r'C:\Users\Administrator\Desktop\Kacper\2020\1\13'
filename='HgCdTe_4734_115K_mala_moc'
frame=read_files(path,filename)
frame=frame.groupby(0, as_index=False).mean()
spectrum,x,y=make_spectrum(frame)

sns.heatmap(np.log(spectrum.iloc[:,180:200]),cmap='hsv')
sns.heatmap(np.log(spectrum_int.iloc[:,160:200]),cmap='hsv')


spectrum_win=spectrum.rolling(50).mean()
sns.heatmap(np.log(spectrum_win.iloc[:,100:300]),cmap='hsv')
sns.heatmap(np.log(spectrum_win.iloc[:,170:200]),cmap='hsv')
plt.plot(np.log(spectrum.loc[:,2312]))
plt.plot(np.log(spectrum.loc[:,2299]))

plt.plot(np.log(spectrum.loc[:,629]+spectrum.loc[:,2719]+spectrum.loc[:,2732]+spectrum.loc[:,2630]))
plt.plot(np.log(spectrum_win.iloc[:,190]))
plt.plot(np.log(spectrum_win.iloc[:,191]))
plt.plot(np.log(spectrum_win.iloc[:,192]))
plt.plot(np.log(spectrum_win.iloc[:,193]))
plt.xlim(100,500)
plt.ylim(-4,0)

y_plot=spectrum_win.loc[:,2288]
x_plot=y[50:-300]
y_plot= (y_plot.values)[50:-300]
x_plot=np.flip(x_plot)

popt,pcov=curve_fit(func, x_plot, y_plot,p0=[0,0,4,4,20,20]
                    ,bounds=([0,0,0,0,0,0],[25,25,200,200,800,800]))
plt.plot(x_plot,y_plot)
plt.plot(x_plot,func(x_plot,*popt))


