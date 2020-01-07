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

def make_spectrum(frame,interp):
    spectrum=fft_frame(frame)
    #spectrum=spectrum.T
    x=interp*np.arange(0,2*15797.76,(2*15797.76/(spectrum.shape[0]))).astype(int)
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
filename='HgCdTe_40K'
frame=read_files(path,filename)
frame=frame.groupby(0, as_index=False).mean()
frame=frame.iloc[1:,1:]
interp=1
frame_int=interp(frame,1/interp)
sns.heatmap(frame.iloc[:,100:500],cmap='hsv') 


spectrum,x,y=make_spectrum(frame,interp)
plt.plot(np.log(spectrum.loc[:,2186]))

x_plot=y[1000:4000]
y_plot= spectrum.loc[25:99.98,2186].values

popt,pcov=curve_fit(func, x_plot, y_plot,p0=[0,40000,20])


spectrum_int=make_spectrum(frame_int,interp)
sns.heatmap(np.log(spectrum.iloc[:,100:500]),cmap='hsv')
spectrum.iloc[:,100:300].to_csv(filename+'FFT1')


spectrum.iloc[100:15000,100:300].to_csv(filename+'FFT')
plt.plot(frame.iloc[1100,10:2000])


#check where starts falling
frame1=frame.iloc[:,400:]

frame=frame.replace([-np.inf],-6)
frame1=frame1-np.min(frame1.min())

plt.plot(np.log(frame.iloc[300,:]))
plt.xlim(200,1000)
plt.plot(np.log(frame.iloc[300,400:]))

spectrum,x,y=make_spectrum_bez_drop(frame,1)
#spectrum1,x1,y1=make_spectrum_bez_drop(frame.iloc[:,:7000],4) za duze
sns.heatmap(spectrum1.iloc[200:2000,10:5000],cmap='hsv')


plt.plot(spectrum1.sum(axis=1))
plt.xlim(15,100)
plt.ylim(0,5000)
spectrum1.isna().sum().sum()
spectrum1=spectrum1.fillna(0)
spectrum1=spectrum1.replace([-np.inf],0)
spectrum1=spectrum1.replace([np.inf],0)

sns.heatmap(np.log(spectrum.T.iloc[800:4000,100:300]),cmap='hsv')
sns.heatmap(spectrum1.iloc[800:8000,120:280],cmap='hsv')
spectrum1.iloc[800:8000,120:280].to_csv(filename+'FFT_HD')
plt.plot(spectrum1.iloc[650:,])
plt.plot(np.log(spectrum1.iloc[650:,180]))
plt.plot(spectrum1.iloc[650:,188])
plt.plot(x1[20:1000]/64,spectrum1.iloc[650,20:1000])

spectrum2=spectrum1.iloc[650:,]
j=15
ica = FastICA(n_components=j)
principalComponents = ica.fit_transform(spectrum2.iloc[:,:].values)
PC=principalComponents
PC1=PC[:,[1,3,7,8,13]]
for i in range(0,j):
    plt.plot(y1[650:],PC[:,14])#0,1,3,6,8,11,13,14
    plt.ylim(-0.05,0.05)
plt.plot(spectrum1.iloc[650:,188]/200)

def grafen_plot(X,a0,a1,a2,a3,a4):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3
        y+=X[:,4]*a4
        return y
    
def col_fit(y):
    popt, pcov = curve_fit(grafen_plot, PC1, y)
    return grafen_plot(PC1, *popt)

spectrum3=spectrum2.apply(lambda x: col_fit(x),axis=0)
sns.heatmap(spectrum3.iloc[:8000,100:600],cmap='hsv')

plt.plot(spectrum3.iloc[:,550])
plt.plot(spectrum3.iloc[:,180])
plt.plot(spectrum3.iloc[:,188])
plt.plot(spectrum3.iloc[:,190])
plt.plot(spectrum2.iloc[:,190])
plt.plot(spectrum3.iloc[0,5:1000])  



