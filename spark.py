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

from pyspark import SparkConf, SparkContext


path =r'C:\Users\Administrator\Desktop\Kacper\2019\11\5'
filename='HgCdTe_40K'

conf=SparkConf().setMaster('local').setAppName('spark')
sc=SparkContext(conf=conf)