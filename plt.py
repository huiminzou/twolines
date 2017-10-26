# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 03:47:15 2017

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
axes.text(lon[-1]-0.1,lat[-1]-0.2,'L1')
a2=-1/a1
lat2=[]
lon2=np.linspace(mlon-0.25,mlon+0.25,20)
for a in np.arange(len(lon2)):
    lat2.append(mlat+a2*(lon2[a]-mlon))
axes.scatter(lon2,lat2)
axes.text(lon2[-1],lat2[-1]+0.2,'L2')
plt.savefig('p1',dpi=300)
deep=np.load('deep1x.npy')
temp=np.load('salinity1x.npy')
np.argmin(np.hstack(deep))
xi = np.arange(lon[0],lon[-1],0.011)
yi = np.arange(0,-180,-5)
x=[]
for a in np.arange(len(lon)):
    for b in np.arange(45):
        x.append(lon[a])
y=np.hstack(deep)
uu=[]
for a in np.arange(len(temp)):
    if a%2==0:
        for b in np.arange(len(temp[a])):
            uu.append(temp[a][b])
uu=np.array(uu)#uu=np.hstack(temp)
plt.show()
fig,axes=plt.subplots(1,1)
xb,yb,ub_mean,ub_median,ub_std,ub_num = sh_bindata(np.array(x), np.array(y), np.array(uu), xi, yi)
xxb,yyb = np.meshgrid(xb, yb)
#cc=np.array([-1., -.75, -.5, -.25, -0.2, -.15, -.1, -0.05, 0., 0.05, .1, .15, .2, .25, .5, .75, 1.])
#fig,axes=plt.subplots(3,2,figsize=(7,5))
#plt.figure()
axes.set_xlabel('longitude')
axes.set_ylabel('depth (m)')
ub1 = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
#vb1 = np.ma.array(vb_mean, mask=np.isnan(vb_mean1))
#ax2 = fig.add_axes()
plt.title('L1')
p = axes.pcolor(xb, yb, ub1.T)#, cmap=matplotlib.cm.RdBu, vmin=((dvf-mvf).T).min(), vmax=((dvf-mvf).T).max())
cb = fig.colorbar(p, ax=axes,label='Degrees Celsius')
#.set_ylabel('degree')
plt.savefig('p2x2008salinity1x',dpi=300)
#Q=plt.quiver(xb,yb,ub1.T,vb1.T,scale=5.)