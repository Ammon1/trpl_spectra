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

def ica_plot7(X,a0,a1,a2,a3,a4,a5,a6):#,a5,a6):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5+X[:,6]*a6
        return y

def ica_plot6(X,a0,a1,a2,a3,a4,a5):#,a5,a6):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5
        return y

def ica_plot5(X,a0,a1,a2,a3,a4):#,a4,a5,a6):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4
        return y
    
def ica_plot4(X,a0,a1,a2,a3):#,a4,a5,a6):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3
        return y
    
def ica_plot3(X,a0,a1,a2):#,a4,a5,a6):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2
        return y

def fit_noise(y,X_transformed):
    bg=y[0]
    y=y-bg
    
    if X_transformed.shape[1]==3:
        popt3, pcov = curve_fit(ica_plot3, X_transformed,y)
        y=ica_plot3(X_transformed, *popt3)+bg
        return np.append(y,popt3)
        
    elif X_transformed.shape[1]==4:
        popt4, pcov = curve_fit(ica_plot4, X_transformed,y)
        y=ica_plot4(X_transformed, *popt4)+bg
        return np.append(y,popt4)
        
    elif X_transformed.shape[1]==5:
        popt5, pcov = curve_fit(ica_plot5, X_transformed,y)
        y=ica_plot5(X_transformed, *popt5)+bg
        return np.append(y,popt5)
    
    elif X_transformed.shape[1]==6:
        popt6, pcov = curve_fit(ica_plot6, X_transformed,y)
        y=ica_plot6(X_transformed, *popt6)+bg
        return np.append(y,popt6)
    
    elif X_transformed.shape[1]==7:
        popt7, pcov = curve_fit(ica_plot7, X_transformed,y)
        y=ica_plot7(X_transformed, *popt7)+bg
        return np.append(y,popt7)
    
    

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


def noise_remover(frame):
        intensities=frame.iloc[:,:].values
        transformer = FastICA(n_components=7,max_iter=20000,
                              fun='cube',
                              random_state=0)
        X_transformed = transformer.fit_transform(intensities)  
        X_transformed=np.delete(transform(X_transformed),0,1)
        print(X_transformed.shape)
        data_ica=frame.apply(lambda y: fit_noise(y,X_transformed))

            
        return data_ica,X_transformed
    
def fft_frame(df):
    x=df.iloc[:,1].values
    y=np.abs(fft(x))
    length=y.shape[0]
    w = blackman(length)
    spectra=df.apply(lambda x: np.abs(fft(w*x)))
    return spectra
    
    
def read_files(path,filename):
    
    frame=pd.read_csv(path+'\\'+filename,index_col=False,sep=' ')
    return frame

def model(x,x_interp,y,interpolate):
    model=interpolate.interp1d(x, y,kind='cubic')
    return model(x_interp)

def interp(df,interpolation):
    x=np.arange(0,df.shape[0])
    x_interp=np.arange(0,x[-1],interpolation)
    df_interp=df.apply(lambda y: model(x,x_interp,y,interpolate))
    return df_interp
    

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

from scipy.signal import savgol_filter   
def time_constant2(x,x0,x1,A0,A1,tau0,tau1):
    return A0*np.exp(-(x-x0)/tau0)+A1*np.exp(-(x-x1)/tau1)

def time_constant1(x,x0,A0,tau0):
    return A0*np.exp(-(x-x0)/tau0)


path =r'C:\Users\Administrator\Desktop\Kacper\2020\8\28'
filename='\\InAs10076'
frame=read_files(path,filename)
frame=frame.groupby(frame.columns.values[0], as_index=False).mean()




spectrum,x,y=make_spectrum(frame)
sns.heatmap(np.log(spectrum.iloc[:100,10:500]),cmap='hsv')
plt.plot(np.log(spectrum.iloc[5,10:600]))
plt.plot(np.log(spectrum.iloc[:,275]))
plt.plot(np.log(spectrum.iloc[0,10:500]))
plt.plot(np.log(spectrum.iloc[5,10:500]))
plt.plot(np.log(spectrum.iloc[10,10:500]))
for i in range(270,290):
    plt.plot(spectrum.iloc[:,i])

spectrum.to_csv('10406_180k.csv')
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
