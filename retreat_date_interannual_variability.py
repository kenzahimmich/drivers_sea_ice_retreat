#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:07:07 2024

@author: Kenza
"""


import xarray as xr
import os
import matplotlib.pyplot as plt

import numpy as np

import warnings

from mpl_toolkits.basemap import Basemap


from scipy import stats
import matplotlib as mpl

from mpl_toolkits.axes_grid1 import make_axes_locatable


warnings.filterwarnings("ignore")

os.chdir('/Users/Kenza/Desktop/drivers_SI_retreat/source_data_code/')
p_indir_base = 'data/'
p_fig =  '../../figures/'

f_yr = 1995
l_yr = 2021
l_obs_yr = list(range(f_yr, l_yr + 1))


tmp_file = xr.open_dataset(p_indir_base + 'dr_sitmax_U_V_yearly_1994-2020_25km.nc', decode_times=True)


dr_conc = tmp_file['dr'].values
uw_conc = tmp_file['U_OND'].values
vw_conc = tmp_file['V_OND'].values
sit_max = tmp_file['sitmax'].values

stdr = tmp_file['std_dr'].values

lons = tmp_file.lon.values
lats = tmp_file.lat.values
    

tmp_file = xr.open_dataset(p_indir_base + 'dr_IRR_U_V_yearly_1994-2020_75km.nc', decode_times=True)

dr_conc75 = tmp_file['dr'].values
uw_conc75 = tmp_file['U_OND'].values
vw_conc75 = tmp_file['V_OND'].values
dsicdr_conc = tmp_file['IRR_OND'].values
resdr_conc = tmp_file['IRR_Res_OND'].values
dyndr_conc = tmp_file['IRR_Dyn_OND'].values

ui_clim = tmp_file['drift_U_OND_clim'].values
vi_clim = tmp_file['drift_V_OND_clim'].values


lons75 = tmp_file.lon.values
lats75 = tmp_file.lat.values


#%% Useful plotting functions


def map_correl(ax,  data, data_c, lons, lats, levels, label,  cmap=mpl.cm.get_cmap('RdBu_r'),cbar=True):

    parallels = [-80, -70, -60.]
    meridians = [-180, -90, 0, 90]
    m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
                resolution='l', boundinglat=-53, ax=ax) 
    xx, yy = m(lons, lats)

    cs = m.contourf(xx, yy, data,
                    levels=levels,
                    cmap=cmap, extend='both',alpha=0.9)

    if data_c != []:
        m.contour(xx, yy, np.where(data_c < 0, 2, 0),
                  colors='black', linewidths=0.08)

    m.fillcontinents(color='lightgrey')
    m.drawcoastlines(linewidth=1, color='lightgrey')
    m.drawparallels(parallels, labels=[
                    True, True, True, True], linewidth=0.5, linestyle=':',  dashes=(2, 5))
    m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.5, linestyle=':',  dashes=(
        2, 5))  

    if cbar == True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size=0.1, pad=0.0)
    
        cbar = plt.colorbar(cs,
                            ax=(ax), cax=cax, orientation="horizontal", extendrect=True)
        cbar.set_label(label)
    else:
        ax.text(0.5, 0.02, '$'+label+'$', transform=ax.transAxes,
                va="bottom", ha="center",color = 'black')
    return cs,m, xx, yy

#%% DETREND 25 KM


years = l_obs_yr[:]

dr_det = np.empty([len(l_obs_yr[:]), 432, 432], dtype='float32')
dr_det[:, :, :] = np.NaN



sit_det = np.empty([len(l_obs_yr[:]),  432, 432], dtype='float32')
sit_det[:, :, :] = np.NaN

vw_det = np.empty([len(l_obs_yr[:]),  432, 432], dtype='float32')
vw_det[:, :, :] = np.NaN


uw_det = np.empty([len(l_obs_yr[:]),  432, 432], dtype='float32')
uw_det[:, :, :] = np.NaN




for yi in range(432):

    for xi in range(432):

        
        X_flat = np.asarray(years)


        Y_flat = dr_conc[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            dr_det[:, yi, xi] = Y_flat - \
                (t_da * X_flat.astype('float32') + inter_da)

        Y_flat = vw_conc[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            vw_det[:, yi, xi] = Y_flat - \
                (t_da * X_flat.astype('float32') + inter_da)
                
        Y_flat = uw_conc[:,  yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            uw_det[:, yi, xi] = Y_flat - \
                (t_da * X_flat.astype('float32') + inter_da)

        
        X_flat = np.asarray(years[:])
        Y_flat = sit_max[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            sit_det[:, yi, xi] = Y_flat - \
                (t_da * X_flat.astype('float32') + inter_da)




#%% DETREND 75 KM

dr_det75 = np.empty([len(l_obs_yr[:]),  144,  144], dtype='float32')
dr_det75[:, :, :] = np.NaN

irr_det = np.empty([len(l_obs_yr[:]),  144,  144], dtype='float32')
irr_det[:, :, :] = np.NaN


vw_det75 = np.empty([len(l_obs_yr[:]),  144,  144], dtype='float32')
vw_det75[:, :, :] = np.NaN

uw_det75 = np.empty([len(l_obs_yr[:]),  144,  144], dtype='float32')
uw_det75[:, :, :] = np.NaN

dyn_det = np.empty([len(l_obs_yr[:]),  144,  144], dtype='float32')
dyn_det[:, :, :] = np.NaN



years = np.asarray(l_obs_yr[:]).astype('float32')


for yi in range(144):

    for xi in range(144):

        X_flat = np.asarray(years)

        Y_flat = dr_conc75[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            dr_det75[:, yi, xi] = Y_flat - \
                (t_da * X_flat + inter_da)

      

        Y_flat = vw_conc75[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):
            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            vw_det75[:, yi, xi] = Y_flat - \
                (t_da * X_flat + inter_da)
                
        
        Y_flat = uw_conc75[:, yi, xi]

        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):
            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            uw_det75[:, yi, xi] = Y_flat - \
                (t_da * X_flat + inter_da)


        Y_flat = dyndr_conc[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            dyn_det[:, yi, xi] = Y_flat - \
                (t_da * X_flat + inter_da)



                
        Y_flat = dsicdr_conc[:, yi, xi]
        if len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years):

            t_da, inter_da, rv, pv, err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            irr_det[:, yi, xi] = Y_flat - \
                (t_da * X_flat + inter_da)

res_det = irr_det - dyn_det  # equivalent`



#%% CORRELATE 25 KM

rvr = np.empty([432, 432], dtype='float32')
rvr[:, :] = np.NaN

pvr = np.empty([432, 432], dtype='float32')
pvr[:, :] = np.NaN

rvu = np.empty([432, 432], dtype='float32')
rvu[:, :] = np.NaN

pvu = np.empty([432, 432], dtype='float32')
pvu[:, :] = np.NaN

rvw = np.empty([432, 432], dtype='float32')
rvw[:, :] = np.NaN

pvw = np.empty([432, 432], dtype='float32')
pvw[:, :] = np.NaN

years = l_obs_yr[:]



for yi in range(432):

    for xi in range(432):

        X_flat = sit_det[:, yi, xi]
        Y_flat = dr_det[:, yi, xi] # 14

        if (len(X_flat[~np.isnan(X_flat)]) > 0.33*len(years)) & (len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years)):
            t, inter, rvr[yi, xi], pvr[yi, xi], err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])

        X_flat = vw_det[:, yi, xi]
        Y_flat = dr_det[:, yi, xi]

        if (len(X_flat[~np.isnan(X_flat)]) > 0.33*len(years)) & (len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years)):
            t, inter, rvw[yi,xi], pvw[yi,xi], err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])

        X_flat = uw_det[:, yi, xi]
        Y_flat = dr_det[:, yi, xi]

        if (len(X_flat[~np.isnan(X_flat)]) > 0.33*len(years)) & (len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years)):
            t, inter, rvu[yi,xi], pvu[yi,xi], err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])




#%% CORRELATE 75 KM


rvu75 = np.empty([144, 144], dtype='float32')
rvu75[:, :] = np.NaN

pvu75 = np.empty([144, 144], dtype='float32')
pvu75[:, :] = np.NaN

rvw75 = np.empty([144, 144], dtype='float32')
rvw75[:, :] = np.NaN

pvw75 = np.empty([144, 144], dtype='float32')
pvw75[:, :] = np.NaN

rirr = np.empty([144,  144], dtype='float32')
rirr[:, :] = np.NaN

pirr = np.empty([144,  144], dtype='float32')
pirr[:, :] = np.NaN

rres = np.empty([144,  144], dtype='float32')
rres[:, :] = np.NaN

pres = np.empty([144,  144], dtype='float32')
pres[:, :] = np.NaN

rdyn = np.empty([144,  144], dtype='float32')
rdyn[:, :] = np.NaN

pdyn = np.empty([144,  144], dtype='float32')
pdyn[:, :] = np.NaN




for yi in range(144):

    for xi in range(144):

        Y_flat = dr_det75[:, yi, xi]
        X_flat = irr_det[:, yi, xi]
            
        if (len(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)]) > 4):
            
            t, inter, rirr[yi, xi], pirr[yi, xi], err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
          
            
        Y_flat = dr_det75[:, yi, xi]
        X_flat = res_det[:,  yi, xi]

        if (len(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)]) > 4) & np.any(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)] != 0):
            t, inter, rres[yi, xi], pres[yi, xi], err=stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])

        Y_flat= dr_det75[:, yi, xi]
        X_flat=dyn_det[:,  yi, xi]

        if (len(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)]) > 4) & np.any(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)] > 0):
            t, inter, rdyn[yi, xi], pdyn[yi, xi], err=stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])
            
            
        

        X_flat = vw_det75[:, yi, xi]
        Y_flat = dr_det75[:, yi, xi]

        if (len(X_flat[~np.isnan(X_flat)]) > 0.33*len(years)) & (len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years)):
            t, inter, rvw75[yi,xi], pvw75[yi,xi], err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])

        X_flat = uw_det75[:, yi, xi]
        Y_flat = dr_det75[:, yi, xi]

        if (len(X_flat[~np.isnan(X_flat)]) > 0.33*len(years)) & (len(Y_flat[~np.isnan(Y_flat)]) > 0.33*len(years)):
            t, inter, rvu75[yi,xi], pvu75[yi,xi], err = stats.linregress(X_flat[~np.isnan(
                X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])

               

#%% CALCULATE PERCENTAGES OF SIGNIFICANT CORRELATION
print('DR_SIT')
sig_drsit = np.count_nonzero(np.where(pvr < 0.05, 1, 0))/np.count_nonzero(np.where(~np.isnan(pvr ), 1, 0))*100
ave_drsit = np.nanmean(np.where(pvr < 0.05, abs(rvr), np.nan) )
std_drsit = np.nanstd(np.where(pvr < 0.05, abs(rvr), np.nan) )
print(sig_drsit)
print( ave_drsit)


print('DR_IRR')
sig_drirr = np.count_nonzero(np.where(pirr < 0.05, 1, 0))/np.count_nonzero(np.where(~np.isnan(pirr ), 1, 0))*100
ave_drirr = np.nanmean(np.where(pirr < 0.05, abs(rirr), np.nan) )
std_drirr = np.nanstd(np.where(pirr < 0.05, abs(rirr), np.nan) )

print(sig_drirr)
print( ave_drirr)


print('DR_RES')
sig_drres = np.count_nonzero(np.where(pres < 0.05, 1, 0))/np.count_nonzero(np.where(~np.isnan(pres ), 1, 0))*100
ave_drres = np.nanmean(np.where(pres < 0.05, abs(rres), np.nan) )
std_drres = np.nanstd(np.where(pres < 0.05, abs(rres), np.nan) )


print(sig_drres)
print( ave_drres)

print('DR_DYN')
sig_drdyn = np.count_nonzero(np.where(pdyn < 0.05, 1, 0))/np.count_nonzero(np.where(~np.isnan(pdyn ), 1, 0))*100
ave_drdyn = np.nanmean(np.where(pdyn < 0.05, abs(rdyn), np.nan) )
std_drdyn = np.nanstd(np.where(pdyn < 0.05, abs(rdyn), np.nan) )


print(sig_drdyn)
print( ave_drdyn)


print('DR_WIND')
sig_drw = np.count_nonzero(np.where((pvw < 0.05)|(pvu < 0.05), 1, 0))/np.count_nonzero(np.where(~np.isnan(pvw ), 1, 0))*100
print(sig_drw)


print('DR_U')
sig_dru = np.count_nonzero(np.where(pvu < 0.05, 1, 0))/np.count_nonzero(np.where(~np.isnan(pvu ), 1, 0))*100
ave_dru = np.nanmean(np.where(pvu < 0.05, abs(rvu), np.nan) )
std_dru = np.nanstd(np.where(pvu < 0.05, abs(rvu), np.nan) )

print(sig_dru)
print( ave_dru)



print('DR_V')
sig_drv = np.count_nonzero(np.where(pvw < 0.05, 1, 0))/np.count_nonzero(np.where(~np.isnan(pvw ), 1, 0))*100
ave_drv = np.nanmean(np.where(pvw < 0.05, abs(rvw), np.nan) )
std_drv = np.nanstd(np.where(pvw < 0.05, abs(rvw), np.nan) )

print(sig_drv)
print( ave_drv)



#%% FIGURE 3 

cm = 1/2.54
fig, axes = plt.subplots(2, 3, figsize=(21*cm, 27.6*cm/1.8),constrained_layout=True) #(210 x 276 mm),

    

rvar=[  rirr, rres, rdyn, rvr, rvu,rvw ]
pvar=[  pirr, pres, pdyn,pvr, pvu,pvw]
labvar = ['r(IRR_{OND},d_r)','r(IRR_{OND}^{Res},d_r)','r(IRR_{OND}^{Dyn},d_r)','r(SIT_{max},d_r)','r(U_{OND},d_r)','r(V_{OND},d_r)']
medvar = [ ave_drirr , ave_drres ,  ave_drdyn ,  ave_drsit ,  ave_dru ,  ave_drv ] 
iqrvar = [  std_drirr ,  std_drres ,   std_drdyn ,   std_drsit ,   std_dru ,   std_drv ] 

levels=np.linspace(-0.8, 0.8, 17)

mi=0
for (i_r, i_p,i_l,imed,iiqr) in zip(rvar, pvar,labvar,medvar,iqrvar):

            
    data= i_r
    data_c=np.where(i_p < 0.05, -1,1)
    lab=i_l
    if mi < 3:
        ax=axes.ravel()[mi]
        
        cs, m, xx, yy=map_correl(ax, -data, data_c, lons75, lats75, levels, lab, cmap = 'RdBu_r',cbar=False)
        m.contourf(xx, yy, np.where(i_p > 0.05, -1,np.nan), colors='aliceblue')
        m.contour(xx, yy, np.where(~np.isnan(i_p), -1,1), colors='lightsteelblue',linewidths=0.1)

        if mi == 1:
            m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
                        resolution='l', boundinglat=-53, ax=ax) #, round=True
   
            xx, yy = m(lons, lats)

            hc=m.contourf(xx, yy, np.where(pvr < 0.05, -1,np.nan), hatches=['....'],colors='none')
            m.contour(xx, yy, np.where(pvr < 0.05, -1,1), colors='black', linewidths=0.1)
            for collection in hc.collections:
                collection.set_edgecolor("black")  # Set hatch color
                collection.set_linewidth(0)      # Adjust hatch thickness



        if mi == 2:
            
            hc=m.contourf(xx, yy, np.where(  (pvu75 < 0.05)|(pvw75 < 0.05), -1, np.nan), hatches=['.....'],colors='none')
            #m.contour(xx, yy, np.where( (pvu75 < 0.05)|(pvw75 < 0.05),-1,1), colors='black', linewidths=0.1)
            for collection in hc.collections:
                collection.set_edgecolor("black")  # Set hatch color
                collection.set_linewidth(0.1)      # Adjust hatch thickness

                
      
        if mi ==0:
            qs = ax.quiver(xx[::3,::3], yy[::3,::3], np.where(~np.isnan(rirr),vi_clim,np.nan)[::3,::3], np.where(~np.isnan(rirr),ui_clim,np.nan)[::3,::3], 
                           width = 0.005,minshaft=2,headwidth=3.5,headlength=3.5,headaxislength=4.5,scale=200,color='black')
     
            plt.quiverkey(qs,0.274, 0.615, 20,'20 km/day',labelpos='S',
                               coordinates='figure',color='black')

        
    elif mi == 3:

        ax=axes.ravel()[mi]

        cs, m, xx, yy=map_correl(ax, data, data_c, lons, lats, levels, lab, cmap = 'RdBu_r',cbar=False)
        m.contourf(xx, yy, np.where(i_p > 0.05, -1,np.nan), colors='aliceblue')
        m.contour(xx, yy, np.where(~np.isnan(i_p), -1,1), colors='lightsteelblue',linewidths=0.1)

    else:

        ax=axes.ravel()[mi]
        cbar=False

        cs,m, xx, yy=map_correl(ax, data, data_c, lons, lats, levels, lab, cmap = 'RdBu_r',cbar=cbar)
        m.contourf(xx, yy, np.where(i_p > 0.05, -1,np.nan), colors='aliceblue')
        m.contour(xx, yy, np.where(~np.isnan(i_p), -1,1), colors='lightsteelblue',linewidths=0.1)
        
    
    
        hc=m.contourf(xx, yy, np.where(  (stdr > 15), -1, np.nan), hatches=['.....'],colors='none')
        for collection in hc.collections:
            collection.set_edgecolor("black")  # Set hatch color
            collection.set_linewidth(0.1)      # Adjust hatch thickness


        if mi + 2 == 6:
            cbar = fig.colorbar(cs,ax=ax,shrink = 1, orientation='horizontal')

    
    ax.text(0.53, 0.48, '$<|r|>=$\n'+str(round(imed,2)) + '$\pm$' + str(round(iiqr,2)),size=8,fontweight='semibold', transform=ax.transAxes,va="bottom", ha="center",color = 'black')
    mi += 1

letvar=['a','b','c','d','e','f']
for (ax,let) in zip(axes.ravel(),letvar):
    ax.text(0.05, 0.95, let,  transform=ax.transAxes, va="top",ha ="left",fontweight = 'bold')     

#plt.savefig(p_fig + 'driver_SI_retreat_Fig_3.png',format = 'png', dpi=300 )


