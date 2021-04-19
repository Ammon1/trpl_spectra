
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

path =r'C:\Users\Administrator\Desktop\Kacper\2020\11\9'
filename='\\#519_200K_kr1'
filename1='\\#519_150K_kr2'
frame=read_files(path,filename)
frame1=read_files(path,filename1)

plt.plot(0.12389*frame.iloc[10:,0].values,frame.iloc[10:,1].values)
plt.plot(0.12389*frame1.iloc[10:,0].values,frame1.iloc[10:,1].values)
plt.xlim(0,500)
plt.xlim(0,500)

