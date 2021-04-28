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
from scipy.fftpack import rfft
from scipy.signal import blackman
from sklearn.svm import SVR


def transform(X_transformed):
    X_transformed_copy=np.zeros(X_transformed.shape[0])
    i=0
    for y in np.transpose(X_transformed):
            rms = np.sqrt(np.mean(y**2)) 
            if np.any(y[y>4*rms]):
                #plt.plot(y)
                #print('ok',i)
                X_transformed_copy=np.vstack((X_transformed_copy,y))
            else:
                pass
                #print('not ok',i)
            i=i+1
    return np.transpose(X_transformed_copy)

#df=np.delete(transform(X_transformed),0,1)

def fft_frame(df):
    x=df.iloc[:,10].values
    y=np.abs(fft(x))
    length=y.shape[0]
    w = blackman(length)
    print(x.shape,' ',y.shape,' ',w.shape)
    plt.plot(fft(w*x))
    spectra=df.apply(lambda x: np.abs(np.fft.fft(w*x)))
    return spectra
    
    
def read_files(path,filename):
    
    frame=pd.read_csv(path+'\\'+filename,index_col=False,sep=' ')
    return frame

def make_spectrum(frame):
    spectrum=fft_frame(frame)
    x=np.arange(0,2*15797.76,(2*15797.76/(spectrum.shape[0]))).astype(int)
    y=np.arange(0,spectrum.shape[1])
    x=np.around(x,decimals=2)
    y=np.around(y,decimals=2)
    spectrum.columns=y
    spectrum=spectrum.set_index(x)
    spectrum=spectrum.T
    sns.heatmap(np.log(spectrum.iloc[:,100:300]),cmap='hsv')
    return spectrum,x,y


path =r'C:\Users\Administrator\Desktop\Kacper\2021\4\28'
filename='\\inas_test10_39.999K_0'#0.145
frame=read_files(path,filename)

#frame=frame.groupby(frame.columns.values[0], as_index=False).mean()


spectrum,x,y=make_spectrum(frame)

sns.heatmap(np.log(spectrum.iloc[:250,100:200]),cmap='hsv')
plt.plot(spectrum.iloc[0,100:200].values)
plt.plot(spectrum.iloc[5,100:200].values)
plt.plot(spectrum.iloc[10,100:200].values)
plt.plot(spectrum.iloc[30,100:200].values)
plt.plot(spectrum.iloc[:,130].values)
plt.plot(spectrum.iloc[:,120].values)

spectrum_win=spectrum.rolling(5).mean()

sns.heatmap(np.log(spectrum_win.iloc[5:50,500:600]),cmap='hsv')

plt.plot(spectrum.iloc[10,120:160].values)
plt.plot(spectrum.iloc[15,120:160].values)
plt.plot(spectrum.iloc[20,120:160].values)
plt.plot(spectrum.iloc[30,120:160].values)
plt.plot(np.log(spectrum.iloc[5:,132].values))

sns.heatmap(np.log(spectrum_win.iloc[5:50,100:200]),cmap='hsv')

#spectrum_ica_win=spectrum_ica.rolling(100).mean()

spectrum_win.iloc[:,100:300].to_csv(path+filename+'TRPL_map')
#sns.heatmap(np.log(spectrum_ica_win.iloc[:,200:250]),cmap='hsv')
sns.heatmap(np.log(spectrum_win.iloc[110:,150:50]),cmap='hsv')

plt.plot(np.log(spectrum_ica1.iloc[:7000,190]))
plt.plot(np.log(spectrum.iloc[:3000,110]))
plt.plot(np.log(spectrum.iloc[:3000,100]))





def fit_time(y_plot):
    y_plot= (y_plot.values)[:]
    x_plot=np.arange(0,2000,2000/y_plot.shape[0])
    #plt.plot(x_plot,y_plot)
    popt2=[0,0,0,0,0,0]
# =============================================================================
#     try:
#         popt1,pcov1=curve_fit(time_constant1, x_plot1, y_plot1,p0=[0,4,2000]
#                     ,bounds=([0,0,0],[25,200,100000]))
#     except:
#         popt1[0,2]=[0,0]
#     #plt.plot(x_plot1,y_plot1)
#     plt.plot(x_plot1,time_constant1(x_plot1,*popt1))
# =============================================================================

    try:
        popt2,pcov2=curve_fit(time_constant2, x_plot, y_plot,p0=[0,0,4,4,10,2000]
                    ,bounds=([0,0,0,0,0,0],[25,25,200,200,200,4000]))
    except:
        print('error')
    #plt.plot(x_plot1,y_plot1)
    #plt.plot(x_plot1,time_constant2(x_plot1,*popt2))
    data=[popt2[0],popt2[1],int(popt2[4]),int(popt2[5])]
    return np.array(data)

spectrum_to_fit=spectrum_ica.iloc[40:,150:250]
data=spectrum_to_fit.apply(lambda y: fit_time(y))
plt.plot(data.iloc[3,:])
plt.xlim(2000,3000)
plt.ylim(1000,3000)
time_170=1750
