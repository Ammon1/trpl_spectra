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

def ica_plot(X,a0,a1,a2,a3,a4,a5,a6):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5+X[:,6]*a6
        return y

def fit_noise(y,X_transformed):
    popt, pcov = curve_fit(ica_plot, X_transformed,y)
    return ica_plot(X_transformed, *popt)

def noise_remover(frame):
        intensities=frame.iloc[:,:].values
        transformer = FastICA(n_components=7,random_state=0)
        X_transformed = transformer.fit_transform(intensities)            
        #intensities_cleaned=np.zeros(intensities.shape)            
        frame_ica=frame.apply(lambda y: fit_noise(y,X_transformed))
        return frame_ica,X_transformed
    
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
def time_constant2(x,x0,x1,A0,A1,tau0,tau1):
    return A0*np.exp(-(x-x0)/tau0)+A1*np.exp(-(x-x1)/tau1)

def time_constant1(x,x0,A0,tau0):
    return A0*np.exp(-(x-x0)/tau0)


path =r'C:\Users\Administrator\Desktop\Kacper\2020\1\14'
filename='HgCdTe_4734_190K_3filtry'
frame=read_files(path,filename)
frame=frame.groupby(0, as_index=False).mean()
plt.plot(frame.iloc[100,40:])

frame_ica,X_transformed=noise_remover(frame.iloc[:,40:])

spectrum_ica,x,y=make_spectrum(frame_ica)
spectrum,x,y=make_spectrum(frame)

sns.heatmap(spectrum.iloc[10:,150:250],cmap='hsv')
sns.heatmap(spectrum_ica.iloc[:,150:250],cmap='hsv')

spectrum_ica_win=spectrum_ica.rolling(100).mean()
spectrum_win=spectrum.rolling(100).mean()
sns.heatmap(np.log(spectrum_ica_win.iloc[:,200:250]),cmap='hsv')
sns.heatmap(np.log(spectrum_win.iloc[110:,200:250]),cmap='hsv')

plt.plot(np.log(spectrum.loc[40:,2285]))
plt.plot(np.log(spectrum_ica.loc[:,2285]))

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
