#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan  6 15:36:35 2024

@author: Kenza
"""


import xarray as xr
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import warnings
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as dplt
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

warnings.filterwarnings("ignore")

os.chdir('/Users/Kenza/Desktop/drivers_SI_retreat/source_data_code/')
p_indir_base = 'data/'
p_fig =  '../../figures/'


tmp_file = xr.open_dataset(p_indir_base + 'dr_sitmax_clim_1994-2020_25km.nc', decode_times=True)
dr_doy_clim = tmp_file['dr_doy_clim'].values
dr_clim = tmp_file['dr_clim'].values

sitmax_obs = tmp_file['sitmax_clim'].values
lons = tmp_file.lon.values
lats = tmp_file.lat.values
    

tmp_file = xr.open_dataset(p_indir_base + 'Res_Dyn_clim_1994-2020_25km.nc', decode_times=True)
res_clim = tmp_file['Res_clim'].values
dyn_clim = tmp_file['Dyn_clim'].values
lons75 = tmp_file.lon.values
lats75 = tmp_file.lat.values

# %% Useful plotting functions


def map_dates(ax,  data, lons, lats, levels, label,  cmap=mpl.cm.get_cmap('RdYlBu_r')):

    parallels = [-80, -70, -60.]
    meridians = [-180, -90, 0, 90]
    data = dplt.date2num(data)

    m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
                resolution='l', boundinglat=-53, ax=ax)  # round=True
    xx, yy = m(lons, lats)


    cs = m.contourf(xx, yy, data,
                    levels=levels,
                    cmap=cmap)

    m.fillcontinents(color='lightgrey')
    m.drawcoastlines(linewidth=1, color='lightgrey')
    m.drawparallels(parallels, labels=[
                    True, True, True, True], linewidth=0.5, linestyle=':',  dashes=(2, 5))
    m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.5, linestyle=':',  dashes=(
        2, 5))  

    # HISTOGRAM

    bins = levels

    divider = make_axes_locatable(ax)
    hs = divider.append_axes('bottom', size='15%', pad=0.05)

    hs.hist(data[~np.isnan(data)], weights=np.zeros_like(data[~np.isnan(data)]) +
            1. / data[~np.isnan(data)].size, bins=bins, color="lightgrey", ec="white")

    hs.xaxis.set_major_locator(dplt.MonthLocator())
    hs.xaxis.set_major_formatter(dplt.DateFormatter('%b'))
    hs.set_ylim(0., 0.3)
    hs.set_xlim(levels[0], levels[-1])

    plt.sca(hs)
    plt.xticks(rotation=70)

    plt.tick_params(labelbottom=False, bottom=False)

    cax = divider.append_axes('bottom', size=0.1, pad=0.0)

    cbar = plt.colorbar(cs,
                        ticks=dplt.MonthLocator(),
                        format=dplt.DateFormatter('%b'),
                        ax=(ax), cax=cax, orientation="horizontal", extendrect=True)

    cbar.ax.set_xticklabels(
        [t.get_text()[0] for t in cbar.ax.get_xticklabels()])

    cbar.set_label(label)

    plt.show()


def map_scalar(ax,  data, lons, lats, levels, label,  cmap=mpl.cm.get_cmap('Spectral_r')):

    parallels = [-80, -70, -60.]
    meridians = [-180, -90, 0, 90]
    m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
                resolution='l', boundinglat=-53, ax=ax)  # , round=True
    xx, yy = m(lons, lats)


    cs = m.contourf(xx, yy, data,
                    levels=levels,
                    cmap=cmap)

    m.fillcontinents(color='lightgrey')
    m.drawcoastlines(linewidth=1, color='lightgrey')
    m.drawparallels(parallels, labels=[
                    True, True, True, True], linewidth=0.5, linestyle=':',  dashes=(2, 5))

    m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.5, linestyle=':',  dashes=(
        2, 5))  

    # HISTOGRAM

    bins = levels

    divider = make_axes_locatable(ax)
    hs = divider.append_axes('bottom', size='15%', pad=0.05)

    hs.hist(data[~np.isnan(data)], weights=np.zeros_like(data[~np.isnan(data)]) +
            1. / data[~np.isnan(data)].size, bins=bins, color="lightgrey", ec="white")

    hs.set_ylim(0., 0.3)
    hs.set_xlim(levels[0], levels[-1])

    plt.sca(hs)
    plt.xticks(rotation=70)
    plt.tick_params(labelbottom=False, bottom=False)

    cax = divider.append_axes('bottom', size=0.1, pad=0.0)

    cbar = plt.colorbar(cs,
                        ax=(ax), cax=cax, orientation="horizontal", extendrect=True)
    cbar.set_label(label)
    return m, xx, yy


def plot_hist2D(ax, X, Y, xlabel, ylabel, xlim1, xlim2, ylim1, ylim2, cmap):

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    t1, inter1, rv1, pv, err1 = stats.linregress(X_flat[~np.isnan(
        X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)])

    print(t1)

    t = t1

    inter = inter1

    h1, xe, ye, im = ax.hist2d(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)], Y_flat[~np.isnan(
        X_flat) & ~np.isnan(Y_flat)], cmap=cmap, bins=50, zorder=-100)

    ax.plot(X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)], t *
            X_flat[~np.isnan(X_flat) & ~np.isnan(Y_flat)] + inter, color='black')

    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(xlim1, xlim2),
           ylim=(ylim1, ylim2), facecolor="#f5faff")

    textstr = '$R^2=%.2f$' + str(round(rv1**2, 2))
    ax.text(0.03, 0.73, textstr, transform=ax.transAxes,
            va="bottom", ha="left", color='black')
    return im


def map_error(ax,  data, data_c, lons, lats, levels, label,  cmap=mpl.cm.get_cmap('RdBu_r')):

    parallels = [-80, -70, -60.]
    meridians = [-180, -90, 0, 90]
    m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
                resolution='l', boundinglat=-53, ax=ax, round=True)
    xx, yy = m(lons, lats)

    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)
    cs = m.contourf(xx, yy, data,
                    levels=levels,
                    cmap=cmap, extend='both')

    m.contour(xx, yy, np.where(data_c < 0, 2, 0),
              colors='black', linewidths=0.2)

    m.fillcontinents(color='lightgrey')
    m.drawcoastlines(linewidth=1, color='lightgrey')
    m.drawparallels(parallels, labels=[
                    True, True, True, True], linewidth=0.5, linestyle=':',  dashes=(2, 5))
    m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.5, linestyle=':',  dashes=(
        2, 5))  

    divider = make_axes_locatable(ax)

    cax = divider.append_axes('bottom', size=0.1, pad=0.0)

    cbar = plt.colorbar(cs,
                        ax=(ax), cax=cax, orientation="horizontal", extendrect=True)
    cbar.set_label(label)


# %% Figure 2


cm = 1/2.54
fig, axes = plt.subplots(1, 3, figsize=(
    21*cm, 27.6*cm/3.2), constrained_layout=True)  # (210 x 276 mm),
fig.tight_layout()


plt.subplots_adjust(right=0.95,
                    left=0.1,
                    bottom=0.15,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.45)


ax = axes.ravel()[0]
ax.text(-0.05, 1.172, 'a',  transform=ax.transAxes,
        va="top", ha="left", fontweight='bold')

levels = np.asarray(np.linspace(dplt.date2num(np.datetime64(
    '1993-09-01')), dplt.date2num(np.datetime64('1994-03-01')), 13))
data = dr_clim
map_dates(ax, data, lons, lats, levels, '$d_r$',
          cmap=mpl.cm.get_cmap('Spectral_r'))


ax = axes.ravel()[1]
ax.text(-0.05, 1.172, 'b',  transform=ax.transAxes,
        va="top", ha="left", fontweight='bold')

levels = np.asarray(np.linspace(0, 4, 17))
data = sitmax_obs
map_scalar(ax, data, lons, lats, levels, '$SIT_{max}$ $[m]$')


ax = axes.ravel()[2]

ax.text(-0.05, 1.08, 'c',  transform=ax.transAxes,
        va="top", ha="left", fontweight='bold')
ax.text(0.05, 0.95, 'All SIZ',  transform=ax.transAxes,
        va="top", ha="left", fontweight='bold')

X = sitmax_obs
Y = dr_doy_clim - 244  # 15 sep
xlabel = '$SIT_{max}$ $[m]$'
ylabel = 'd$_r$ (days)'
xlim1 = 0
xlim2 = 5
ylim1 = 0
ylim2 = 200
cmap = 'Blues'
im = plot_hist2D(ax, X, Y, xlabel, ylabel, xlim1, xlim2, ylim1, ylim2, cmap)
cbar_ax = inset_axes(ax, width="50%", height="3%", loc='lower right',
                     borderpad=1.8)  # Customize size and location
cbar = fig.colorbar(mappable=im, ax=ax, cax=cbar_ax, orientation='horizontal')

cbar.set_label('# grid points', labelpad=-36, y=-10, rotation=0, color='grey')

#plt.savefig(p_fig + 'driver_SI_retreat_Fig_2.png',format = 'png', dpi=300 )


#%% Figure 1 

thermo = res_clim
dyna = dyn_clim


cm = 1/2.54
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
    21*cm*0.7, 27.6*cm/3.35), constrained_layout=True)  # (210 x 276 mm),


ax1.text(0.05, 0.95, 'a',  transform=ax1.transAxes,
         va="top", ha="left", fontweight='bold')
ax2.text(0.05, 0.95, 'b',  transform=ax2.transAxes,
         va="top", ha="left", fontweight='bold')



parallels = [-80, -70, -60.]
meridians = [-180, -90, 0, 90]


# DYNAMCIS

data = 100*dyna

m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
            resolution='l', boundinglat=-52.5, ax=ax1)

xx, yy = m(lons75, lats75)


cs = m.contourf(xx, yy, data,
                levels=np.linspace(-200, 200, 21),
                cmap='RdBu')


m.contour(xx, yy, data,
          levels=np.linspace(-200, 200, 21), extend='both', linewidths=0.08,
          colors='black')
cont_dyn = np.where((dyna < 0) & (thermo != 0), abs(dyna/(thermo)), np.nan)
m.contour(xx, yy, np.where(cont_dyn > 1, -1, 1),
          levels=np.linspace(-2, 2, 21), extend='both', linewidths=0.4,
          colors='black')


cmap = plt.cm.get_cmap("RdBu")
bounds = np.asarray(np.linspace(-200, 200, 21))
norm = mpl.colors.Normalize(vmin=-200, vmax=200)
ax1.pcolormesh(xx, yy, data, norm=norm, cmap=cmap)

m.fillcontinents(color='lightgrey')
m.drawcoastlines(linewidth=1, color='lightgrey')
m.drawparallels(parallels, labels=[
                True, True, True, True], linewidth=0.5, linestyle=':',  dashes=(2, 5))
m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.5,
                linestyle=':',  dashes=(2, 5))  # labels = [left,right,top,bottom]

ax1.text(0.5, 0.03, 'Dyn',  transform=ax1.transAxes, va="bottom", ha="center")


# THERMODYNAMICS

data = 100*(thermo)


m = Basemap(projection='spaeqd', lat_0=-90, lon_0=180,
            resolution='l', boundinglat=-52.5, ax=ax2)


cs = m.contourf(xx, yy, data,
                levels=np.linspace(-200, 200, 21),
                cmap='RdBu')


m.contour(xx, yy, data,
          levels=np.linspace(-200, 200, 21), extend='both', linewidths=0.08,
          colors='black')


cmap = plt.cm.get_cmap("RdBu")
bounds = np.asarray(np.linspace(-200, 200, 21))
norm = mpl.colors.Normalize(vmin=-200, vmax=200)
ax2.pcolormesh(xx, yy, data, norm=norm, cmap=cmap)


m.fillcontinents(color='lightgrey')
m.drawcoastlines(linewidth=1, color='lightgrey')
m.drawparallels(parallels, labels=[
                True, True, True, True], linewidth=0.5, linestyle=':',  dashes=(2, 5))
m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.5,
                linestyle=':',  dashes=(2, 5))  # labels = [left,right,top,bottom]


ax2.text(0.5, 0.03, 'Res',  transform=ax2.transAxes, va="bottom", ha="center")


cbar = fig.colorbar(cs,
                    ax=(ax1,ax2), shrink=0.5,  aspect=10, orientation="horizontal", extendrect=True)
cbar.set_label('[%]', labelpad=1, y=-10, rotation=0)


#plt.savefig(p_fig + 'driver_SI_retreat_Fig_1.png', format='png', dpi=300)


#%% CALCULATION OF DYN-DRIVEN AND MELT-DRIVEN ZONES AREA

data = np.where( (thermo != 0), abs((dyna)/thermo), np.nan) 
print(np.nanmin(data))

print('% of dynamic ice removal')
print(len(data[(abs(data) > 1) & (dyna < 0)])/len(data[~np.isnan(data)]))
print(len(data[(abs(data) < 1) ])/len(data[~np.isnan(data)]))


