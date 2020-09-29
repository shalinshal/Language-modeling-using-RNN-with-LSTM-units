# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:33:05 2019

@author: Shalin
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#task 1
#x = np.arange(0,20)
x_ticks = [1, 1, 0, 4, 3, 4, 0, 2, 0, 2, 0, 4, 3, 0, 2, 4, 5, 0, 9, 0, 4]

for filename in glob.glob('*1.csv'):
    if filename[:1]=='f':
        f_1=pd.read_csv(filename)
    if filename[:1]=='g':
        g_1=pd.read_csv(filename)
    if filename[:1]=='i':
        i_1=pd.read_csv(filename)
    if filename[:1]=='o':
        o_1=pd.read_csv(filename)
    if filename[:1]=='p':
        p_1=pd.read_csv(filename)

#task 2
for filename in glob.glob('*2.csv'):
    if filename[:1]=='f':
        f_2=pd.read_csv(filename)
    if filename[:1]=='g':
        g_2=pd.read_csv(filename)
    if filename[:1]=='i':
        i_2=pd.read_csv(filename)
    if filename[:1]=='o':
        o_2=pd.read_csv(filename)
    if filename[:1]=='p':
        p_2=pd.read_csv(filename)
plt.figure()
f_2_0=f_2['0']
f_2_2=f_2['1']
o_2_0=o_2['0']
o_2_2=o_2['1']
plt.xticks(range(20),x_ticks)
plt.plot(f_2_0)
plt.plot(f_2_2)
plt.plot(o_2_0)
plt.plot(o_2_2)
plt.legend(['Gate F-0','Gate F-2','Gate O-0','Gate O-2'])
plt.xlabel('Input Sequence')
plt.ylabel('Gate value')
plt.show()
plt.savefig('Task 2-F O.png')

plt.figure()
g_2_0=g_2['0']
g_2_2=g_2['1']
i_2_0=i_2['0']
i_2_2=i_2['1']
plt.xticks(range(20),x_ticks)
plt.plot(g_2_0)
plt.plot(g_2_2)
plt.plot(i_2_0)
plt.plot(i_2_2)
plt.legend(['Gate G-0','Gate G-2','Gate I-0','Gate I-2'])
plt.xlabel('Input Sequence')
plt.ylabel('Gate value')
plt.show()
plt.savefig('Task 2-G I.png')

#task 3
for filename in glob.glob('*3.csv'):
    if filename[:1]=='f':
        f_3=pd.read_csv(filename)
    if filename[:1]=='g':
        g_3=pd.read_csv(filename)
    if filename[:1]=='i':
        i_3=pd.read_csv(filename)
    if filename[:1]=='o':
        o_3=pd.read_csv(filename)
    if filename[:1]=='p':
        p_3=pd.read_csv(filename)
plt.figure()
f_3_0=f_3['0']
f_3_2=f_3['1']
o_3_0=o_3['0']
o_3_2=o_3['1']
plt.xticks(range(20),x_ticks)
plt.plot(f_3_0)
plt.plot(f_3_2)
plt.plot(o_3_0)
plt.plot(o_3_2)
plt.legend(['Gate F-0','Gate F-2','Gate O-0','Gate O-2'])
plt.xlabel('Input Sequence')
plt.ylabel('Gate value')
plt.show()
plt.savefig('Task 3-F O.png')

plt.figure()
g_3_0=g_3['0']
g_3_2=g_3['1']
i_3_0=i_3['0']
i_3_2=i_3['1']
plt.xticks(range(20),x_ticks)
plt.plot(g_3_0)
plt.plot(g_3_2)
plt.plot(i_3_0)
plt.plot(i_3_2)
plt.legend(['Gate G-0','Gate G-2','Gate I-0','Gate I-2'])
plt.xlabel('Input Sequence')
plt.ylabel('Gate value')
plt.show()
plt.savefig('Task 3-G I.png')