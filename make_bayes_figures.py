#!/usr/bin/env python3
"""Bayesian-inference figures: modelled vs observed OSL erosion rates, posteriors, present-day erosion map.

Usage: python make_bayes_figures.py <output.nc> <bayes.json> <outdir> [tag]
"""
import json, sys, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()
from matplotlib.colors import LogNorm
from erosion_params import (ref, hillslope_mm_yr, hillslope_range, GLACIAL_RANGE,
                            PLOT_D_MM_YR, PLOT_SC_DEG)
from mapdeco import graticule, scalebar
from pyproj import Transformer
from trunk_swath import swath_mask

OSL='data_osl/osl_erosion_rates.csv'


def bed_slope(z,dx,dy):
    gy,gx=np.gradient(z,dy,dx); return np.hypot(gx,gy)


def main(nc,js,outdir,tag=''):
    os.makedirs(outdir,exist_ok=True)
    R=json.load(open(js))['results']
    ds=xr.open_dataset(nc)
    x,y=ds.x.values,ds.y.values
    dx,dy=abs(float(x[1]-x[0])),abs(float(y[1]-y[0])); cell=dx*dy/1e6
    topg=ds.topg.values; topg=topg[0] if topg.ndim==3 else topg
    thk=ds.thk.values; ub=ds.velbase_mag.values; t=ds.time.values; ds.close()
    sw,tr,dkm,cat=swath_mask(topg,x,y)
    S=bed_slope(topg,dx,dy); ice=thk>1
    w=np.gradient(t); w=w/w.sum()
    from osl_sites import load_sites, describe
    osl=load_sites()          # surveyed coordinates, not Table 1's ~1 km rounding
    print('  '+describe(osl))
    trf=Transformer.from_crs('EPSG:4326','EPSG:32759',always_xy=True)
    ux,uy=trf.transform(osl.longitude.values,osl.latitude.values)
    cells=[(int(np.argmin(np.abs(y-sy))),int(np.argmin(np.abs(x-sx))))
           for sx,sy in zip(ux,uy)]

    def predict(l,kg,D,sc_deg):
        sc=np.tan(np.deg2rad(sc_deg))
        hill=D*S/(1-np.clip(S/sc,0,0.999)**2)
        tot=np.zeros_like(topg)
        for i in range(len(t)):
            tot+=w[i]*np.where(ice[i],kg*ub[i]**l,hill)
        out=[]
        for cy,cx in cells:
            out.append(tot[max(cy-1,0):cy+2,max(cx-1,0):cx+2].mean()*1e3)
        return np.array(out),tot

    obs=osl.erosion_rate.values
    err=[osl.sigma_minus.values,osl.sigma_plus.values]
    preds={}
    for l in ('1','2'):
        if l not in R: continue
        r=R[l]
        p,tot=predict(int(l),r['Kg']['median'],PLOT_D_MM_YR*1e-3,PLOT_SC_DEG)
        preds[l]=(p,tot,r)

    fig,ax=plt.subplots(figsize=(6.8,6.6))
    mk={'1':('o','tab:blue'),'2':('s','tab:red')}
    allv=[obs-err[0], obs+err[1]]+[p for p,_,_ in preds.values()]
    lo=min(float(np.min(v)) for v in allv); hi=max(float(np.max(v)) for v in allv)
    lim=[lo/1.6, hi*1.6]
    for l,(p,_,r) in preds.items():
        ax.errorbar(obs,p,xerr=err,fmt=mk[l][0],color=mk[l][1],ms=7,capsize=3,
                    mec='k',mew=.6,alpha=.85,
                    label=f"$l$={l}  ($K_g$={r['Kg']['median']:.2e})")
    ax.plot(lim,lim,'k-',lw=1.2,label='1:1')
    ax.plot(lim,[v*2 for v in lim],'k--',lw=.8,label='factor of 2')
    ax.plot(lim,[v/2 for v in lim],'k--',lw=.8)
    ax.set(xscale='log',yscale='log',xlim=lim,ylim=lim,
           xlabel='OSL-inferred erosion rate (mm yr$^{-1}$)',
           ylabel='modelled erosion rate (mm yr$^{-1}$)')

    from collections import defaultdict
    bycell=defaultdict(list)
    for i,nm in enumerate(osl['sample']):
        bycell[cells[i]].append((i,nm))
    for grp in bycell.values():
        if len(grp) < 2: continue
        idx=[i for i,_ in grp]
        for l,(p,_,_) in preds.items():
            ax.annotate(' + '.join(nm[-3:] for _,nm in grp),
                        (float(max(obs[i] for i in idx)), p[idx[0]]),
                        xytext=(13,0), textcoords='offset points',
                        ha='left', va='center', fontsize=7.5, color=mk[l][1])
    ax.legend(loc='lower right'); ax.grid(alpha=.3,which='both')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig6_osl_predicted_vs_observed.png',dpi=150)
    plt.close(fig); print(f'wrote {outdir}/fig6_osl_predicted_vs_observed.png')

    fig=plt.figure(figsize=(14.0,7.6))
    gs=fig.add_gridspec(2,3,hspace=.42,wspace=.30)
    axa=fig.add_subplot(gs[:,0:2])
    for l,(p,_,r) in preds.items():
        axa.errorbar(obs,p,xerr=err,fmt=mk[l][0],color=mk[l][1],ms=7,capsize=3,
                     mec='k',mew=.6,label=f'$l$={l}')
    axa.plot(lim,lim,'k-',lw=1.2); axa.plot(lim,[v*2 for v in lim],'k--',lw=.8)
    axa.plot(lim,[v/2 for v in lim],'k--',lw=.8)
    figstyle.panel(axa,'a')
    axa.set(xscale='log',yscale='log',xlim=lim,ylim=lim,
            xlabel='OSL-inferred (mm yr$^{-1}$)',ylabel='modelled (mm yr$^{-1}$)')
    axa.legend(); axa.grid(alpha=.3,which='both')
    slots={0:gs[0,2]}
    for col,(nm,unit,logsc) in enumerate([('Kg','m yr$^{-1}$ per (m yr$^{-1}$)$^l$',True)]):
        a1=fig.add_subplot(slots[col])
        figstyle.panel(a1,col+1)
        for l,(p,_,r) in preds.items():
            q=r[nm]
            v=[q['lo68'],q['median'],q['hi68']]
            if nm=='D': v=[z*1e3 for z in v]
            a1.errorbar([int(l)],[v[1]],yerr=[[v[1]-v[0]],[v[2]-v[1]]],
                        fmt=mk[l][0],color=mk[l][1],ms=9,capsize=5,mec='k',mew=.6)
            a1.annotate(f"{v[1]:.3g}",(int(l),v[1]),xytext=(9,0),
                        textcoords='offset points',fontsize=8,va='center')
        if logsc: a1.set_yscale('log')
        a1.set(xticks=[1,2],xlabel='exponent $l$',ylabel=f'{nm} ({unit})',
               xlim=(0.6,2.6))
        a1.grid(alpha=.3)
    axs=fig.add_subplot(gs[1,2])
    axs.axis('off')
    rows=[]
    for l,(p,_,r) in preds.items():
        rows.append(f"$l$={l}:  $K_g$={r['Kg']['median']:.3g} "
                    f"[{r['Kg']['lo68']:.3g}, {r['Kg']['hi68']:.3g}]   "
                    f"$D$={PLOT_D_MM_YR:g} mm/yr (fixed)   "
                    f"$S_c$={PLOT_SC_DEG:g}$^\\circ$ (fixed)")
        rows.append(f"      $\\chi^2$/site = {r['chi2_min_per_site']:.2f},  "
                    f"ESS = {r['ESS']:.0f},  median model/obs = "
                    f"{np.median(p/obs):.2f}")
    axs.text(-0.15,.98,"Posterior medians and 68% CIs\n\n"+"\n".join(
                 r.replace('   ','\n      ') for r in rows),
             va='top',ha='left',fontsize=7.6,family='monospace')
    fig.savefig(f'{outdir}/fig7_bayesian_inference.png',dpi=150,bbox_inches='tight')
    plt.close(fig); print(f'wrote {outdir}/fig7_bayesian_inference.png')

    i = -1
    ys,xs=np.where(ice[i]|cat)
    ny,nx=topg.shape
    sl=(slice(max(ys.min()-6,0),min(ys.max()+7,ny)),
        slice(max(xs.min()-6,0),min(xs.max()+7,nx)))
    ls=sorted(preds)
    fig=plt.figure(figsize=(8.0*len(ls),6.6))
    gs=fig.add_gridspec(2,len(ls),height_ratios=[1,.05],hspace=.04,wspace=.10,
                        left=.045,right=.985,top=.965,bottom=.105)
    axes=[[fig.add_subplot(gs[0,k]) for k in range(len(ls))]]
    handles={}
    for ax,l in zip(axes[0],ls):
        rr=preds[l][2]
        par=ref(int(l),kg=rr['Kg']['median'],D=PLOT_D_MM_YR,Sc=PLOT_SC_DEG)
        hrange=hillslope_range(hillslope_mm_yr(S,par['D'],par['Sc']))
        hill=hillslope_mm_yr(S,par['D'],par['Sc'],floor=hrange[0])
        gl=par['Kg']*ub[i]**int(l)*1e3
        ax.imshow(topg[sl],cmap='gray',origin='lower',alpha=.85)
        ih=ax.imshow(np.where(ice[i],np.nan,hill)[sl],cmap='YlOrRd',origin='lower',
                     norm=LogNorm(*hrange))
        im=ax.imshow(np.where(ice[i],gl,np.nan)[sl],cmap='Blues',origin='lower',
                     norm=LogNorm(*GLACIAL_RANGE))
        ax.contour(ice[i][sl].astype(float),[.5],colors='#08306b',linewidths=1.0)
        ax.contour(cat[sl].astype(float),[.5],colors='k',linewidths=1.1,linestyles='--')
        ax.text(.03, .965, f'$l$={int(l)}\n$K_g$={par["Kg"]:.2e}\n$D$={par["D"]:.2e} mm/yr\n'
                f'$S_c$={par["Sc"]:.1f}$^\\circ$', transform=ax.transAxes, va='top', ha='left',
                fontsize=mpl.rcParams['xtick.labelsize']*.92,
                bbox=dict(fc='white', ec='none', alpha=.80, pad=3))
        figstyle.panel(ax, list(axes[0]).index(ax))
        ax.set_xticks([]); ax.set_yticks([])
        graticule(ax,x[sl[1]],y[sl[0]],nlines=2,fontsize=7.5)
        scalebar(ax,dx,fontsize=8)
        handles[l]=(im,ih)
    cax_g=fig.add_subplot(gs[1,0]); cax_h=fig.add_subplot(gs[1,1])
    cb=fig.colorbar(handles[ls[0]][0],cax=cax_g,orientation='horizontal')
    cb.set_label('glacial erosion (mm yr$^{-1}$, log)')
    cb=fig.colorbar(handles[ls[0]][1],cax=cax_h,orientation='horizontal')
    cb.set_label('hillslope erosion (mm yr$^{-1}$, log)')
    fig.savefig(f'{outdir}/fig8_present_day_erosion.png',dpi=150)
    plt.close(fig); print(f'wrote {outdir}/fig8_present_day_erosion.png')


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4] if len(sys.argv)>4 else '')
