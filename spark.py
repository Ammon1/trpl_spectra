import pandas as pd
import numpy as np
#from pyspark import SparkConf, SparkContext
#from pyspark.sql.types import StructField, StructType, IntegerType,FloatType, StringType, TimestampType
from databricks import koalas as ks
from pyspark.sql import SparkSession

from scipy.fftpack import fft
from scipy.fftpack import rfft
from scipy.signal import blackman
import seaborn as sns
from scipy import interpolate

def model(x,x_interp,y,interpolate):
    print('start2.1')
    model=interpolate.interp1d(x, y,kind='cubic')
    return model(x_interp)

def interp(df,shape,interpolation):
    print('start')
    x=np.arange(0,shape)
    print('start1')
    x_interp=np.arange(0,x[-1],interpolation)
    print('start2')
    df_interp=df.T.applymap(lambda y: model(x,x_interp,y,interpolate))
    print('start3')
    return df_interp
    
def fft_frame(df,shape):
    print('funkcja2')
    length=shape
    print('shape')
    w = blackman(length)
    print('blackman')
    spectra=df.T.applymap(lambda y:y+2)#df.applymap(lambda x: np.abs(fft(w*x)))
    print('fft')
    return spectra

def make_spectrum(frame,shape):
    print('funckja')
    spectrum=fft_frame(frame,shape)
    x=np.arange(0,2*15797.76,(2*15797.76/(spectrum.shape[0]))).astype(int)
    y=np.arange(0,spectrum.shape[1])
    x=np.around(x,decimals=2)
    y=np.around(y,decimals=2)
    spectrum.columns=y
    spectrum=spectrum.set_index(x)
    spectrum=spectrum.T
    sns.heatmap(np.log(spectrum.iloc[:,100:300]),cmap='hsv')
    return spectrum,x,y

#path =r'C:\Users\Administrator\Desktop\Kacper\2019\11\5'
filename='\HgCdTe_4734_1'

pdf=pd.read_csv(filename)
print('resising')
pdf=pdf*10000
print('resized')
kdf = ks.from_pandas(pdf.astype(int))
print('to int')
kdf=kdf.groupby(kdf.columns.values[0], as_index=False).mean()
print('interpolation')
#spectrum,x,y=make_spectrum(kdf,pdf.shape[0])
frame_int=interp(kdf,pdf.shape[0],0.5)
#conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
#sc = SparkContext(conf = conf)
#rdd_mapped = sc.textFile("file:///Users\Administrator\Desktop\Kacper\2019\11\5"+filename)

#schema = StructType([FloatType])

#df = rdd_mapped.toDF(schema)


#frame=frame.groupby('1.000', as_index=False).mean()