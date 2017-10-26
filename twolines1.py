# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 01:32:31 2017

@author: xiaojian
"""
import numpy as np
#from pydap.client import open_url
import matplotlib.pyplot as plt
#from SeaHorseLib import *
from datetime import *

from matplotlib.path import Path
#from scipy import interpolate
#import sys
#from SeaHorseTide import *
#import shutil
from netCDF4 import Dataset
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from datetime import datetime, timedelta
def sh_bindata(x, y, z, xbins, ybins):
    """
    Bin irregularly spaced data on a rectangular grid.

    """
    ix=np.digitize(x,xbins)
    iy=np.digitize(y,ybins)
    xb=0.5*(xbins[:-1]+xbins[1:]) # bin x centers
    yb=0.5*(ybins[:-1]+ybins[1:]) # bin y centers
    zb_mean=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_median=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_std=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_num=np.zeros((len(xbins)-1,len(ybins)-1),dtype=int)    
    for iix in range(1,len(xbins)):
        for iiy in range(1,len(ybins)):
#            k=np.where((ix==iix) and (iy==iiy)) # wrong syntax
            k,=np.where((ix==iix) & (iy==iiy))
            zb_mean[iix-1,iiy-1]=np.mean(z[k])
            zb_median[iix-1,iiy-1]=np.median(z[k])
            zb_std[iix-1,iiy-1]=np.std(z[k])
            zb_num[iix-1,iiy-1]=len(z[k])
            
    return xb,yb,zb_mean,zb_median,zb_std,zb_num

FNCL='necscoast_worldvec.dat'
CL=np.genfromtxt(FNCL,names=['lon','lat'])
fig,axes=plt.subplots(1,1,figsize=(6,6))#figure()
axes.plot(CL['lon'],CL['lat'])
axes.axis([-67,-65.5,44,45.5])
length=60*0.009009009
latc=np.linspace(44.65,45.02,10)
lonc=np.linspace(-66.6,-65.93,10)

p1 = Path.circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),radius=length)

cl=plt.Circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),length,alpha=0.6,color='yellow')
axes.add_patch(cl)
mlon=(lonc[5]+lonc[4])/2
mlat=(latc[5]+latc[4])/2
axes.scatter(mlon,mlat)
lon1=-65.8
lat1=45.15
a1=(lat1-mlat)/(lon1-mlon)
lon=np.linspace(-66.8,-65.75,100)
lat=[]
for a in np.arange(len(lon)):
    lat.append(mlat+a1*(lon[a]-mlon))
axes.scatter(lon,lat)

a2=-1/a1
lat2=[]
lon2=np.linspace(mlon-0.25,mlon+0.25,20)
for a in np.arange(len(lon2)):
    lat2.append(mlat+a2*(lon2[a]-mlon))
axes.scatter(lon2,lat2)
url='''http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?h[{1}],siglay[0:1:44][{1}],temp[{0}:1:{2}][0:1:44][{1}],salinity[{0}:1:{2}][0:1:44][{1}]'''
#ds = Dataset(url,'r').variables
urltime='''http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?time[0:1:342347]'''
ds = Dataset(urltime,'r').variables
latf=np.load('gom3.lat.npy')
lonf=np.load('gom3.lon.npy')
index1=[]
for a in np.arange(len(lon)):
    d=[]
    for b in np.arange(len(lonf)):
        d.append((lonf[b]-lon[a])*(lonf[b]-lon[a])+(latf[b]-lat[a])*(latf[b]-lat[a]))
    index1.append(np.argmin(d))
index2=[]
for a in np.arange(len(lon2)):
    d=[]
    for b in np.arange(len(lonf)):
        d.append((lonf[b]-lon2[a])*(lonf[b]-lon2[a])+(latf[b]-lat2[a])*(latf[b]-lat2[a]))
    index2.append(np.argmin(d))


date1=datetime(2009,7,14)
t1=datetime(1858,11,17,00,00,00)
time1=[]

for a in np.arange(len(ds['time'][:])):
    print a
    time1.append(t1+timedelta(hours=ds['time'][:][a]*24))
indext=[]
indext.append(np.argmin(abs(np.array(time1)-date1)))
date2=datetime(2009,8,13)
indext.append(np.argmin(abs(np.array(time1)-date2)))

#indext=[268014,268735]
deep=[]
temp=[]
for a in np.arange(len(lon)):
    print a
    deep0=[]
    temp0=[]
    tt=[]
    url1 = url.format(indext[0], index1[a],indext[1])
    print index1[a]
    ds1 = Dataset(url1,'r').variables
    for b1 in np.arange(len(ds1['siglay'])):
        deep0.append(ds1['siglay'][b1][0]*ds1['h'][0])
        print ds1['h'][:]
    for b in np.arange(len(ds1['salinity'][:])):
        print 'b',b
        temp1=[]
        for b2 in np.arange(len(ds1['salinity'][0])):
            temp1.append(ds1['salinity'][b][b2][0])
        temp0.append(temp1)
    xxx=np.array(temp0).T
    for a1 in np.arange(len(xxx)):
        print 'a1',a1
        tt.append(np.mean(xxx[a1]))
    temp.append(tt)
    deep.append(deep0)
    temp.append(temp0)
np.save('deep1x1',deep)
np.save('salinity1x1',temp)
"""
lat=[]
lon=np.linspace(-66.8,-65.75,20)
for a in np.arange(10):
    lat.append(mlat-(10-(a+1))*0.03)
for a in np.arange(10):
    lat.append(mlat+(a+1)*0.03)
axes.scatter(lon,lat)
"""
"""
lon1=np.linspace(-66.8,-65.7,20)
lon1=np.linspace(-66.8,-65.7,20)
"""