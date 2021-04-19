# -*- coding: utf-8 -*
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import mysql.connector
from mysql.connector import Error

def lin(x,A,B):
    return A*x+B

def lin_fit(time,y):
    popt, pcov = curve_fit(lin,time,y)
    print(popt)
    return popt

def connect_write(data):
    try:
        connection = mysql.connector.connect(host='127.0.0.1',
                                             database='TRPL',
                                             user='root',
                                             password='root')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
    
    except Exception as e:
            print("Error while connecting to MySQL", e)
    
    try:
        mySql_insert_query ="""INSERT INTO TRPL.InAsSb
        (sample,wave,gain,filters,power,temperature,intensity,tau)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"""
    
        cursor = connection.cursor()
        cursor.execute(mySql_insert_query,data)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Laptop table")
        cursor.close()
        
    except Exception as e:
        print(e)


##land and prepare
path =r'C:\Users\Administrator\Desktop\Kacper\2020\7\2'
filename='\\10251_2.4_20_1_0.8_300_100.csv'
########wektor danych
sample=10251
wave=2.25
gain=2.2
filters='0'
power=0.82
temperature=110
###########
frame=pd.read_csv(path+filename,sep=' ',header=None).T
T=frame.iloc[0]
time=np.arange(1,frame.shape[0])
frame=frame.iloc[1:]
########wykres i okreslenie granic
plt.plot(time[80:400],frame.iloc[80:400].values)
plt.yscale('log')

up=200
down_int=80
down=80
###dopasowanie
intensity=(np.max(frame.iloc[down_int:].values)-
                np.min(frame.iloc[down_int:].values))

frame=frame-np.min(frame.iloc[down:].values)
yy=np.log(np.array(frame.iloc[down:up].values))
time1=time[down:up]
popt=lin_fit(time1,yy[:,0])
####weryfikacjay
plt.plot(time1,lin(time1,*popt))
plt.plot(time1,yy[:,0])
tau=-1/popt[0]

##probka/dl.fali/wzmocnienie/filtry/moc_opo/temperatura/intensywnosc/czas zaniku
data_point=(sample,wave,gain,filters,power,temperature,intensity,tau)
connect_write(data_point)

tau=73
tau=60
tau=55

        
