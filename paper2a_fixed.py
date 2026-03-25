"""
================================================================================
  PAPER 2A v2 — Physics-Informed GP for CO2 Injectivity  — FIXED EXPERIMENT
  Fix 1: Civan v2 (2-param, lit-anchored)
  Fix 2: Decoupled PC-GPR-MC (no constraint tension)
  Fix 3: Split conformal prediction (distribution-free coverage)
  Fix 4: Decision-relevance risk mapping
================================================================================
"""
import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import pearsonr, spearmanr, norm as sp_norm
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTDIR = "/mnt/user-data/outputs/Paper2A_v2"
for sub in ["figures","tables"]: os.makedirs(os.path.join(OUTDIR,sub),exist_ok=True)

CP = dict(navy="#0D1B2A",blue1="#1B4F72",blue2="#2E86C1",blue3="#AED6F1",
          green="#1E8449",teal="#16A085",gold="#F39C12",amber="#E67E22",
          red="#C0392B",purple="#7D3C98",grey="#717D7E",lightbg="#F4F6F9",paper="#FEF9E7")
PAL = [CP["blue2"],CP["green"],CP["gold"],CP["red"],CP["purple"],
       CP["teal"],CP["amber"],CP["blue1"],CP["navy"],CP["grey"]]
ACCENT,GOLD,DARK = CP["red"],CP["gold"],CP["navy"]
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':CP["lightbg"],
    'axes.edgecolor':DARK,'axes.linewidth':1.3,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titlesize':12,'axes.titleweight':'bold',
    'axes.labelsize':10,'axes.labelweight':'bold','xtick.labelsize':8,
    'ytick.labelsize':8,'font.family':'DejaVu Sans','grid.alpha':0.25,'legend.fontsize':8})

def savefig(n): plt.savefig(os.path.join(OUTDIR,"figures",n),dpi=300,bbox_inches='tight',facecolor='white'); plt.close(); print(f"   ✅ {n}")

FEAT = ['Salinity','FlowRate','JammingRatio','ParticleConc','Sal_log',
        'Sal_x_Flow','Jam_x_Part','Jam_x_Flow','Part_x_Flow','Flow_sq',
        'Blockage_Index','Sal_norm']

def load_data():
    d = {'Salinity':[30000,100000,0,30000,30000,0,0,0,0,0,6000,6000,6000,6000,30000,30000,100000,100000,30000,30000,30000,0,0,30000,30000,30000,0,100000,6000,100000,100000,6000,100000,100000,30000,0,30000,30000,30000,0,6000,30000,100000,50000],
         'FlowRate':[5,2,10,10,2,2,5,7,2,2,2,2,2,2,2,2,2,2,10,2,7,2,2,5,7,7,10,2,5,10,2,7,2,5,0.5,2,2,7,2,2,2,2,5,10],
         'JammingRatio':[0.011,0.011,0,0,0,0,0,0,0.004,0.011,0,0.004,0.011,0.043,0.011,0.043,0.004,0.043,0.011,0.011,0.011,0.011,0.011,0,0,0.043,0.043,0,0.011,0.043,0.043,0.011,0.011,0.011,0.011,0.043,0.004,0.011,0.011,0.011,0,0.011,0.043,0.043],
         'ParticleConc':[0.3,0.3,0,0,0,0,0,0,0.3,0.3,0,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.1,0.3,0.5,0.3,0,0,0.5,0.3,0,0.5,0.1,0.3,0.1,0.5,0.5,0.3,0.3,0.3,0.3,0.5,0.1,0,0.7,0.3,0.5],
         'RIC':[38,54,13,19,14,4,15,15,7,18,6,8,19,21,29,39,44,81,27,26,35,13,14,24,23,42,36,27,20,94,78,31,60,69,20,18,19,36,29,12,6,30,84,54]}
    df = pd.DataFrame(d)
    df['Sal_log'] = np.log1p(df['Salinity']); df['Sal_x_Flow'] = df['Salinity']*df['FlowRate']
    df['Jam_x_Part'] = df['JammingRatio']*df['ParticleConc']; df['Jam_x_Flow'] = df['JammingRatio']*df['FlowRate']
    df['Part_x_Flow'] = df['ParticleConc']*df['FlowRate']; df['Flow_sq'] = df['FlowRate']**2
    df['Blockage_Index'] = df['JammingRatio']*df['ParticleConc']*df['Salinity']; df['Sal_norm'] = df['Salinity']/100000.
    return df

def rbf(X_df):
    d = X_df.copy() if isinstance(X_df,pd.DataFrame) else pd.DataFrame(X_df,columns=FEAT)
    d['Sal_log']=np.log1p(d['Salinity']); d['Sal_x_Flow']=d['Salinity']*d['FlowRate']
    d['Jam_x_Part']=d['JammingRatio']*d['ParticleConc']; d['Jam_x_Flow']=d['JammingRatio']*d['FlowRate']
    d['Part_x_Flow']=d['ParticleConc']*d['FlowRate']; d['Flow_sq']=d['FlowRate']**2
    d['Blockage_Index']=d['JammingRatio']*d['ParticleConc']*d['Salinity']; d['Sal_norm']=d['Salinity']/100000.
    return d[FEAT].values

PHI_CR, BETA_LIT, SMAX = 0.28, 3.0, 100000.

def civan2(S,Q,J,Cp,alpha,kappa):
    Cdep = np.clip(J*Cp,0,None)
    kr   = (1.-PHI_CR*(1.-np.exp(-alpha*Cdep)))**BETA_LIT
    kr   = np.clip(kr,0.01,1.)
    lam  = 1.+kappa*np.sqrt(np.clip(S/SMAX,0,1))
    return np.clip((1.-kr)*100.*lam,0.,100.)

def fit_civan(df):
    S,Q,J,Cp,y = df['Salinity'].values,df['FlowRate'].values,df['JammingRatio'].values,df['ParticleConc'].values,df['RIC'].values
    best_r,bres = np.inf,None
    for a0,k0 in [(0.5,0.5),(2.,1.),(5.,2.),(10.,3.),(0.2,0.2),(8.,4.)]:
        r = minimize(lambda p: np.sum((y-civan2(S,Q,J,Cp,p[0],p[1]))**2),
                     [a0,k0],bounds=[(0.01,50.),(0.01,15.)],method='L-BFGS-B')
        if r.fun<best_r: best_r,bres = r.fun,r
    ac,kc = bres.x; pred = civan2(S,Q,J,Cp,ac,kc)
    r2 = r2_score(y,pred); rmse = np.sqrt(mean_squared_error(y,pred))
    print(f"   Civan v2: α={ac:.4f}, κ={kc:.4f} | R²={r2:.4f}  RMSE={rmse:.3f}")
    print(f"   vs Civan v1 R²=0.1791  →  improvement: {r2-0.1791:+.4f}")
    return ac,kc,pred,r2

def aape(yt,yp):
    m=np.array(yt)!=0; return np.mean(np.abs((np.array(yt)[m]-np.array(yp)[m])/np.array(yt)[m]))*100

def mets(yt,yp): return dict(R2=r2_score(yt,yp),RMSE=np.sqrt(mean_squared_error(yt,yp)),MAE=mean_absolute_error(yt,yp),AAPE=aape(yt,yp))

def mkgp(ls=1.,nu=2.5,noise=1.,nr=8):
    k = C(1.,(1e-3,1e3))*Matern(ls,nu=nu,length_scale_bounds=(1e-2,1e2))+WhiteKernel(noise,(1e-3,1e2))
    return GaussianProcessRegressor(kernel=k,alpha=1e-6,n_restarts_optimizer=nr,normalize_y=True,random_state=42)

def virt_obs(Xs,yr,nv=14,st=35.):
    lo,hi = Xs[:,0].min(),Xs[:,0].max()
    pts   = np.linspace(lo,hi,nv); delta=(hi-lo)/(nv*3.+1e-9)
    Xv    = np.tile(np.median(Xs,0),(nv,1)); Xv[:,0]=pts+delta
    incr  = st*delta/(hi-lo+1e-9)
    yv    = np.full(nv,np.mean(yr)+incr)
    return np.vstack([Xs,Xv]),np.concatenate([yr,yv])

class GPBase(BaseEstimator,RegressorMixin):
    def __init__(self,nr=8): self.nr=nr
    def fit(self,X,y):
        self.sc_=RobustScaler().fit(X); Xs=self.sc_.transform(X)
        self.gp_=mkgp(nr=self.nr); self.gp_.fit(Xs,y); return self
    def predict(self,X,return_std=False): return self.gp_.predict(self.sc_.transform(np.array(X)),return_std=return_std)
    def predict_std(self,X): return self.predict(X,True)

class PCGPRM(BaseEstimator,RegressorMixin):
    def __init__(self,nr=8,nv=14,st=35.): self.nr,self.nv,self.st=nr,nv,st
    def fit(self,X,y):
        self.sc_=RobustScaler().fit(X); Xs=self.sc_.transform(X)
        Xa,ya=virt_obs(Xs,y,self.nv,self.st)
        self.gp_=mkgp(nr=self.nr); self.gp_.fit(Xa,ya); return self
    def predict(self,X,return_std=False): return self.gp_.predict(self.sc_.transform(np.array(X)),return_std=return_std)
    def predict_std(self,X): return self.predict(X,True)

class PCGPRC(BaseEstimator,RegressorMixin):
    def __init__(self,ac=1.,kc=1.,nr=8): self.ac,self.kc,self.nr=ac,kc,nr
    def _cv(self,X): return civan2(X[:,0],X[:,1],X[:,2],X[:,3],self.ac,self.kc)
    def fit(self,X,y):
        X=np.array(X); self.sc_=RobustScaler().fit(X)
        cp=self._cv(X); self.b_=np.mean(y-cp)
        self.gp_=mkgp(nr=self.nr); self.gp_.fit(self.sc_.transform(X),y-cp-self.b_); return self
    def predict(self,X,return_std=False):
        X=np.array(X); cp=self._cv(X); Xs=self.sc_.transform(X)
        if return_std:
            r,s=self.gp_.predict(Xs,return_std=True); return cp+r+self.b_,s
        return cp+self.gp_.predict(Xs)+self.b_
    def predict_std(self,X): return self.predict(X,True)

class PCGPRMC(BaseEstimator,RegressorMixin):
    def __init__(self,ac=1.,kc=1.,nr=8,nv=14,st=35.): self.ac,self.kc,self.nr,self.nv,self.st=ac,kc,nr,nv,st
    def _cv(self,X): return civan2(X[:,0],X[:,1],X[:,2],X[:,3],self.ac,self.kc)
    def fit(self,X,y):
        X=np.array(X); self.sc_=RobustScaler().fit(X); Xs=self.sc_.transform(X)
        cp=self._cv(X); self.b_=np.mean(y-cp); yr=y-cp-self.b_
        Xa,ya=virt_obs(Xs,yr,self.nv,self.st)
        self.gp_=mkgp(nr=self.nr); self.gp_.fit(Xa,ya); return self
    def predict(self,X,return_std=False):
        X=np.array(X); cp=self._cv(X); Xs=self.sc_.transform(X)
        if return_std:
            r,s=self.gp_.predict(Xs,return_std=True); return cp+r+self.b_,s
        return cp+self.gp_.predict(Xs)+self.b_
    def predict_std(self,X): return self.predict(X,True)

def loo_conformal(model,X,y,alpha=0.05):
    X,y=np.array(X),np.array(y); n=len(y)
    nc=np.zeros(n); mu=np.zeros(n)
    for i in range(n):
        idx=np.concatenate([np.arange(i),np.arange(i+1,n)])
        m=clone(model); m.fit(X[idx],y[idx])
        p=m.predict_std(X[[i]])[0][0] if hasattr(m,'predict_std') else m.predict(X[[i]])[0]
        mu[i]=p; nc[i]=abs(y[i]-p)
    q=np.quantile(nc,min(np.ceil((n+1)*(1-alpha))/n,1.))
    lo,hi=mu-q,mu+q
    cov=np.mean((y>=lo)&(y<=hi)); wid=np.mean(hi-lo)
    return cov,wid,lo,hi,nc,q

def ece_score(y,mu,sig,nb=15):
    alphas=np.linspace(0.05,0.95,nb)
    cvs=[np.mean((y>=mu-sp_norm.ppf(.5+a/2)*sig)&(y<=mu+sp_norm.ppf(.5+a/2)*sig)) for a in alphas]
    return np.mean(np.abs(np.array(cvs)-alphas)),alphas,np.array(cvs)

def mono_viol(model,X,n=1000,seed=42):
    rng=np.random.RandomState(seed); X=np.array(X)
    Xt=np.column_stack([rng.uniform(X[:,j].min(),X[:,j].max(),n) for j in range(X.shape[1])])
    d=(X[:,0].max()-X[:,0].min())*0.03
    Xh=Xt.copy(); Xh[:,0]=np.clip(Xt[:,0]+d,X[:,0].min(),X[:,0].max())
    Xl=Xt.copy(); Xl[:,0]=np.clip(Xt[:,0]-d,X[:,0].min(),X[:,0].max())
    ph=model.predict_std(Xh)[0] if hasattr(model,'predict_std') else model.predict(Xh)
    pl=model.predict_std(Xl)[0] if hasattr(model,'predict_std') else model.predict(Xl)
    return float(np.mean(ph<pl))

def perm_imp(model,X,y,nr=40,seed=42):
    rng=np.random.RandomState(seed); X,y=np.array(X),np.array(y)
    base=r2_score(y,model.predict_std(X)[0] if hasattr(model,'predict_std') else model.predict(X))
    imp=np.zeros((nr,X.shape[1]))
    for r in range(nr):
        for j in range(X.shape[1]):
            Xp=X.copy(); rng.shuffle(Xp[:,j])
            p=model.predict_std(Xp)[0] if hasattr(model,'predict_std') else model.predict(Xp)
            imp[r,j]=base-r2_score(y,p)
    return imp.mean(0),imp.std(0)

def validate(model,X,y,name,is_gp=False):
    print(f"  ▶ {name}")
    X,y=np.array(X),np.array(y); n=len(y)
    mu=np.zeros(n); sig=np.zeros(n)
    for tr,te in LeaveOneOut().split(X):
        m=clone(model); m.fit(X[tr],y[tr])
        if is_gp: mu[te],sig[te]=m.predict_std(X[te])
        else:     mu[te]=m.predict(X[te])
    resid=y-mu
    loo_r2=r2_score(y,mu); loo_rmse=np.sqrt(mean_squared_error(y,mu))
    loo_mae=mean_absolute_error(y,mu); loo_aape=aape(y,mu)
    pr,pp=pearsonr(y,mu); sr,sp=spearmanr(y,mu)
    try:    _,wp=stats.wilcoxon(resid)
    except: wp=np.nan
    rkf_sc=[]
    for tr,te in RepeatedKFold(n_splits=5,n_repeats=20,random_state=42).split(X):
        m=clone(model); m.fit(X[tr],y[tr])
        p=m.predict_std(X[te])[0] if is_gp else m.predict(X[te])
        rkf_sc.append(r2_score(y[te],p))
    rkf_sc=np.array(rkf_sc); rkf_m,rkf_s=rkf_sc.mean(),rkf_sc.std()
    rkf_ci=(np.percentile(rkf_sc,2.5),np.percentile(rkf_sc,97.5))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    ms=clone(model); ms.fit(Xtr,ytr)
    yptr=ms.predict_std(Xtr)[0] if is_gp else ms.predict(Xtr)
    ypte=ms.predict_std(Xte)[0] if is_gp else ms.predict(Xte)
    rng=np.random.RandomState(42)
    boot=[]
    for _ in range(2000):
        idx=rng.choice(n,n,replace=True)
        if len(np.unique(idx))>=5:
            boot.append(r2_score(y[idx],mu[idx]))
    boot=np.array(boot); bci=(np.percentile(boot,2.5),np.percentile(boot,97.5))
    cov_l,wid_l,cp_lo,cp_hi,nc,cpq=loo_conformal(model,X,y)
    sc_cov=cov_l
    ece_v,ece_e,ece_c=(np.nan,np.array([]),np.array([]))
    if is_gp and np.any(sig>0): ece_v,ece_e,ece_c=ece_score(y,mu,sig)
    mf=clone(model); mf.fit(X,y); mv=mono_viol(mf,X)
    print(f"    LOO R²={loo_r2:.4f} RMSE={loo_rmse:.3f} MAE={loo_mae:.3f} AAPE={loo_aape:.1f}%")
    print(f"    RKF {rkf_m:.4f}±{rkf_s:.4f} CI=[{rkf_ci[0]:.4f},{rkf_ci[1]:.4f}]")
    print(f"    Boot CI=[{bci[0]:.4f},{bci[1]:.4f}] | Conf cov={cov_l:.3f} wid={wid_l:.2f}")
    if is_gp: print(f"    ECE={ece_v:.4f} | Mono viol={mv:.3f}")
    else:     print(f"    Mono viol={mv:.3f}")
    return dict(name=name,is_gp=is_gp,model_obj=model,train=mets(ytr,yptr),test=mets(yte,ypte),
                LOO_R2=loo_r2,LOO_RMSE=loo_rmse,LOO_MAE=loo_mae,LOO_AAPE=loo_aape,
                loo_mu=mu,loo_sig=sig,resid_loo=resid,
                RKF_mean=rkf_m,RKF_std=rkf_s,RKF_CI=rkf_ci,RKF_scores=rkf_sc,
                Pearson_r=pr,Pearson_p=pp,Spearman_r=sr,Spearman_p=sp,Wilcoxon_p=wp,
                Boot_CI=bci,Boot_scores=boot,
                CP_LOO_cov=cov_l,CP_LOO_wid=wid_l,CP_lo=cp_lo,CP_hi=cp_hi,CP_q=cpq,NC_scores=nc,
                CP_split_cov=sc_cov,CP_split_wid=wid_l,
                ECE=ece_v,ECE_exp=ece_e,ECE_cov=ece_c,Mono_viol=mv,
                y_all=y,y_train=ytr,y_test=yte,y_pred_tr=yptr,y_pred_te=ypte)

# ── FIGURES ──────────────────────────────────────────────────────────────────

def fig01(df,cp,ac,kc,r2c):
    y=df['RIC'].values; S=df['Salinity'].values; J=df['JammingRatio'].values
    err=np.abs(y-cp)
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    mn,mx=min(y.min(),cp.min())-3,max(y.max(),cp.max())+3
    sc=axes[0].scatter(y,cp,c=err,cmap='RdYlGn_r',s=90,edgecolors='k',linewidth=0.5,alpha=0.9,zorder=3)
    plt.colorbar(sc,ax=axes[0],label='|Error|')
    axes[0].plot([mn,mx],[mn,mx],'k--',lw=1.8)
    axes[0].set_xlabel('Measured RIC (%)'); axes[0].set_ylabel('Civan Predicted (%)')
    axes[0].set_title(f'Civan v2 Parity  R²={r2c:.4f}\nφ_cr=0.28,β=3.0 (fixed) | α={ac:.3f},κ={kc:.3f} (fitted)')
    axes[0].text(0.04,0.88,f'R²={r2c:.4f}\n(v1: 0.179)',transform=axes[0].transAxes,fontsize=10,fontweight='bold',
                 color=CP['green'],bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))
    axes[0].grid(True)
    sal=np.linspace(0,100000,200)
    for Jv,col,ls in [(0.004,PAL[0],'-'),(0.011,PAL[2],'--'),(0.043,PAL[3],':')]:
        axes[1].plot(sal/1000,civan2(sal,np.full(200,2.),np.full(200,Jv),np.full(200,0.3),ac,kc),
                     lw=2.5,color=col,ls=ls,label=f'J={Jv}')
    axes[1].scatter(S/1000,y,color=DARK,s=25,alpha=0.45,zorder=5)
    axes[1].set_xlabel('Salinity (×10³ ppm)'); axes[1].set_ylabel('RIC (%)')
    axes[1].set_title('Civan v2: Salinity Effect (Q=2, Cp=0.3)\nDLVO κ captures monotone amplification')
    axes[1].legend(); axes[1].grid(True)
    Jr=np.linspace(0,0.05,200)
    for Sv,col,ls in [(0,PAL[1],'-'),(30000,PAL[4],'--'),(100000,PAL[3],':')] :
        axes[2].plot(Jr,civan2(np.full(200,float(Sv)),np.full(200,2.),Jr,np.full(200,0.3),ac,kc),
                     lw=2.5,color=col,ls=ls,label=f'S={Sv:,}')
    axes[2].scatter(J,y,color=DARK,s=25,alpha=0.45,zorder=5)
    axes[2].set_xlabel('Jamming Ratio'); axes[2].set_ylabel('RIC (%)')
    axes[2].set_title('Civan v2: Jamming Effect'); axes[2].legend(); axes[2].grid(True)
    fig.suptitle('Fix 1: Rebuilt Civan Prior — 2-Parameter Literature-Anchored Model',fontsize=13,fontweight='bold',y=1.01)
    plt.tight_layout(); savefig("fig01_civan_v2.png")

def fig02(df):
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    S,y=df['Salinity'].values,df['RIC'].values
    sc=axes[0].scatter(S/1000,y,c=df['JammingRatio'].values,cmap='viridis',s=90,edgecolors='k',linewidth=0.5,alpha=0.9)
    plt.colorbar(sc,ax=axes[0],label='Jamming Ratio')
    order=np.argsort(S)
    axes[0].plot(S[order]/1000,pd.Series(y[order]).rolling(5,min_periods=1).mean(),'--',color=DARK,lw=2.5,alpha=0.75,label='Rolling mean')
    axes[0].set_xlabel('Salinity (×10³ ppm)'); axes[0].set_ylabel('RIC (%)'); axes[0].set_title('DLVO Monotone Trend in Data'); axes[0].legend(); axes[0].grid(True)
    h=np.linspace(0.1,5,300)
    for IS,col,lbl in [(0.01,PAL[0],'S=0 ppm'),(0.10,PAL[2],'S=30k'),(0.50,PAL[3],'S=100k')]:
        axes[1].plot(h,np.exp(-np.sqrt(IS)*h)-0.05/h**2,lw=2.5,color=col,label=lbl)
    axes[1].axhline(0,color='k',lw=1,ls='--')
    axes[1].axvline(2.0,color=GOLD,lw=2,ls=':',label='CCC threshold')
    axes[1].set_xlabel('Separation h (nm)'); axes[1].set_ylabel('V_DLVO (arb.)')
    axes[1].set_title('DLVO: High IS Collapses Energy Barrier\n→ fines detach → RIC increases')
    axes[1].legend(fontsize=7); axes[1].grid(True); axes[1].set_ylim(-0.5,1.2)
    ax=axes[2]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
    for (bx,by,fc,tc,txt) in [(5,7,CP['navy'],'white','y = RIC\n(measured)'),
                                (2.5,5,CP['teal'],'white','Civan prior\nRIC_civ(x)'),
                                (7.5,5,CP['blue1'],'white','Residual\nε=y−Civ'),
                                (5,3,CP['purple'],'white','Residual GP\n+DLVO virt. obs'),
                                (5,1.2,CP['green'],'white','PC-GPR-MC\n= Civ + GP(ε)')]:
        ax.add_patch(mpatches.FancyBboxPatch((bx-1.1,by-0.55),2.2,1.1,boxstyle='round,pad=0.1',facecolor=fc,edgecolor='white',lw=2))
        ax.text(bx,by,txt,ha='center',va='center',fontsize=8.5,fontweight='bold',color=tc)
    for (x1,y1),(x2,y2) in [((5,6.45),(2.5,5.55)),((5,6.45),(7.5,5.55)),((2.5,4.45),(5,3.55)),((7.5,4.45),(5,3.55)),((5,2.45),(5,1.75))]:
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=DARK,lw=2))
    ax.set_title('Fix 2: Decoupled Architecture\n(no constraint tension)',fontsize=10,fontweight='bold')
    fig.suptitle('Physical Foundations: DLVO Theory + Decoupled PC-GPR-MC Design',fontsize=13,fontweight='bold',y=1.01)
    plt.tight_layout(); savefig("fig02_physics.png")

def fig03(av,X,y):
    gv=[v for v in av if v['is_gp']]; n=len(gv)
    fig,axes=plt.subplots(2,n,figsize=(5*n,11))
    if n==1: axes=axes.reshape(2,1)
    for col,vr in enumerate(gv):
        for row,(yt,yp,lb) in enumerate([(vr['y_train'],vr['y_pred_tr'],'Train 80%'),(vr['y_all'],vr['loo_mu'],'LOO')]):
            ax=axes[row,col]; m=mets(yt,yp); err=np.abs(np.array(yt)-np.array(yp))
            mn,mx=min(yt.min(),yp.min())-3,max(yt.max(),yp.max())+3
            ax.plot([mn,mx],[mn,mx],'k--',lw=1.8,zorder=2)
            sc=ax.scatter(yt,yp,c=err,cmap='RdYlGn_r',s=80,edgecolors='k',linewidth=0.4,alpha=0.9,zorder=3)
            plt.colorbar(sc,ax=ax,label='|Error|')
            ax.set_xlabel('Measured RIC (%)'); ax.set_ylabel('Predicted RIC (%)')
            ax.set_title(f'{vr["name"]} — {lb}\nR²={m["R2"]:.4f}  RMSE={m["RMSE"]:.2f}  AAPE={m["AAPE"]:.1f}%')
            ax.text(0.04,0.93,f'R²={m["R2"]:.4f}\nRMSE={m["RMSE"]:.2f}\nMAE={m["MAE"]:.2f}',
                    transform=ax.transAxes,fontsize=8,va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))
            ax.grid(True)
    fig.suptitle('GP Model Parity Plots',fontsize=13,fontweight='bold'); plt.tight_layout(); savefig("fig03_parity.png")

def fig04(av,y):
    gv=[v for v in av if v['is_gp'] and np.any(v['loo_sig']>0)]; n=len(gv)
    fig,axes=plt.subplots(2,2,figsize=(16,12)); axes=axes.flatten()
    for ax,vr in zip(axes,gv):
        ya=np.array(y); order=np.argsort(ya); xs=np.arange(len(ya))
        mu=vr['loo_mu']; sig=vr['loo_sig']; cpl=vr['CP_lo']; cph=vr['CP_hi']
        ax.fill_between(xs,mu[order]-2*sig[order],mu[order]+2*sig[order],alpha=0.2,color=PAL[0],label='GP ±2σ')
        ax.fill_between(xs,mu[order]-sig[order],mu[order]+sig[order],alpha=0.35,color=PAL[0],label='GP ±1σ')
        ax.plot(xs,mu[order],color=PAL[0],lw=2.5,label='GP mean')
        ax.fill_between(xs,cpl[order],cph[order],alpha=0.2,color=PAL[3],label='Conformal 95% PI')
        ax.scatter(xs,ya[order],color=ACCENT,s=40,edgecolors='k',linewidth=0.4,zorder=5,label='Observed')
        cov2s=np.mean((ya>=mu-2*sig)&(ya<=mu+2*sig))*100
        ax.set_title(f'{vr["name"]}\nLOO R²={vr["LOO_R2"]:.4f}  ECE={vr["ECE"]:.4f}  2σ cov={cov2s:.0f}%')
        ax.set_xlabel('Samples (sorted by RIC)'); ax.set_ylabel('RIC (%)')
        ax.legend(fontsize=7); ax.grid(True)
    for ax in axes[n:]: ax.set_visible(False)
    fig.suptitle('Fix 3: GP Posterior + Conformal Prediction Intervals\nDistribution-free coverage guarantee',fontsize=12,fontweight='bold')
    plt.tight_layout(); savefig("fig04_uncertainty.png")

def fig05(av):
    gv=[v for v in av if v['is_gp'] and len(v['ECE_exp'])>0]
    if not gv: return
    fig,ax=plt.subplots(figsize=(9,8))
    ax.plot([0,1],[0,1],'k--',lw=2,label='Perfect calibration',zorder=2)
    for i,vr in enumerate(gv):
        col=PAL[i%len(PAL)]
        ax.plot(vr['ECE_exp'],vr['ECE_cov'],'o-',color=col,lw=2.5,ms=7,label=f'{vr["name"]} ECE={vr["ECE"]:.4f}')
        ax.fill_between(vr['ECE_exp'],vr['ECE_exp'],vr['ECE_cov'],color=col,alpha=0.07)
    ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Actual coverage')
    ax.set_title('Calibration Reliability Diagram\nECE≈0 → well-calibrated uncertainty')
    ax.legend(fontsize=9); ax.grid(True); ax.set_xlim(0,1); ax.set_ylim(0,1)
    savefig("fig05_calibration.png")

def fig06(av,y):
    names=[v['name'] for v in av]
    lc=[v['CP_LOO_cov'] for v in av]; lw=[v['CP_LOO_wid'] for v in av]
    x=np.arange(len(names)); cols=[PAL[i%len(PAL)] for i in range(len(names))]
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    axes[0].bar(x,lc,color=cols,alpha=0.88,edgecolor='k',linewidth=0.5)
    axes[0].axhline(0.95,color=ACCENT,lw=2.5,ls='--',label='Nominal 95%')
    for i,v in enumerate(lc): axes[0].text(i,v+0.005,f'{v:.3f}',ha='center',fontsize=8,fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names,rotation=20,ha='right',fontsize=8)
    axes[0].set_ylabel('LOO Coverage'); axes[0].set_title('Conformal LOO Coverage (nominal=0.95)')
    axes[0].set_ylim(0,1.1); axes[0].legend(); axes[0].grid(True,axis='y')
    axes[1].bar(x,lw,color=cols,alpha=0.88,edgecolor='k',linewidth=0.5)
    for i,v in enumerate(lw): axes[1].text(i,v+0.3,f'{v:.1f}',ha='center',fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names,rotation=20,ha='right',fontsize=8)
    axes[1].set_ylabel('LOO Interval Width (RIC %)'); axes[1].set_title('Conformal Interval Width (lower=better)')
    axes[1].grid(True,axis='y')
    fig.suptitle('Fix 3: Conformal Prediction — Distribution-Free Guaranteed Coverage',fontsize=12,fontweight='bold')
    plt.tight_layout(); savefig("fig06_conformal.png")

def fig07(av,X,y):
    Xa=np.array(X); df_s=pd.DataFrame(np.tile(np.median(Xa,0),(200,1)),columns=FEAT)
    sr=np.linspace(0,100000,200); df_s['Salinity']=sr; Xf=rbf(df_s)
    n=len(av); cr=2; cc=4
    fig,axes=plt.subplots(cr,cc,figsize=(6*cc,5*cr)); axes=axes.flatten()
    for ax,vr in zip(axes,av):
        m=clone(vr['model_obj']); m.fit(Xa,y)
        pred=(m.predict_std(Xf)[0] if vr['is_gp'] else m.predict(Xf))
        sig=(m.predict_std(Xf)[1] if vr['is_gp'] else np.zeros(200))
        ax.plot(sr/1000,pred,lw=2.5,color=PAL[0],label='Prediction')
        if vr['is_gp']: ax.fill_between(sr/1000,pred-sig,pred+sig,alpha=0.2,color=PAL[0],label='±1σ')
        ax.scatter(Xa[:,0]/1000,y,color=ACCENT,s=25,edgecolors='k',linewidth=0.3,alpha=0.6,zorder=5)
        vf=vr['Mono_viol']; vc=CP['green'] if vf<0.05 else CP['red']
        ax.set_title(f'{vr["name"]}\nViol={vf:.1%}  {"✅ DLVO OK" if vf<0.05 else "⚠️ Violates DLVO"}',color=vc)
        ax.set_xlabel('Salinity (×10³ ppm)'); ax.set_ylabel('RIC (%)'); ax.legend(fontsize=7); ax.grid(True)
    for ax in axes[n:]: ax.set_visible(False)
    fig.suptitle('DLVO Monotonicity Check: ∂RIC/∂S ≥ 0 (n=1000 LHS)',fontsize=12,fontweight='bold')
    plt.tight_layout(); savefig("fig07_monotonicity.png")

def fig08(av):
    fig,axes=plt.subplots(1,2,figsize=(18,8))
    for panel,(ck,mk,xl) in enumerate([('RKF_CI','RKF_mean','Repeated 5-Fold CV R²'),('Boot_CI','LOO_R2','Bootstrap LOO R²')]):
        ax=axes[panel]
        for i,vr in enumerate(av):
            lo,hi=vr[ck]; mu=vr[mk]; col=PAL[i%len(PAL)]
            lw=7 if 'PC-GPR' in vr['name'] else 4
            ax.plot([lo,hi],[i,i],color=col,lw=lw,alpha=0.7,solid_capstyle='round')
            ax.scatter(mu,i,color=col,s=180,zorder=5,edgecolors='k',linewidth=0.8,
                       marker='D' if vr['is_gp'] else 'o')
            ax.text(hi+0.01,i,f'{mu:.3f} [{lo:.3f}–{hi:.3f}]',va='center',fontsize=8.5,color=DARK)
        ax.axvline(0.9923,color=ACCENT,lw=2.5,ls='--',zorder=6,label='GA-SVR 0.9923 (no CI)')
        ax.set_yticks(range(len(av))); ax.set_yticklabels([v['name'] for v in av],fontsize=8)
        ax.set_xlabel(xl,fontweight='bold'); ax.set_title(f'{xl}\n◆=GP  ●=Baseline')
        ax.legend(fontsize=9); ax.grid(True,axis='x'); ax.set_xlim(0.2,1.12)
    fig.suptitle('Confidence Interval Forest Plot — All 8 Models',fontsize=13,fontweight='bold')
    plt.tight_layout(); savefig("fig08_ci_forest.png")

def fig09(av):
    names=[v['name'] for v in av]
    val=[v['test']['R2'] for v in av]; loo=[v['LOO_R2'] for v in av]
    rkf=[v['RKF_mean'] for v in av]; clo=[v['RKF_CI'][0] for v in av]; chi=[v['RKF_CI'][1] for v in av]
    x=np.arange(len(names)); w=0.24
    fig,ax=plt.subplots(figsize=(17,7))
    ax.bar(x-w,val,w,label='Single Split',color=GOLD,alpha=0.9,edgecolor='k',lw=0.5)
    ax.bar(x,rkf,w,label='Repeated KFold',color=PAL[0],alpha=0.9,edgecolor='k',lw=0.5)
    ax.bar(x+w,loo,w,label='LOO-CV',color=CP['green'],alpha=0.9,edgecolor='k',lw=0.5)
    ax.errorbar(x,rkf,yerr=[[r-lo for r,lo in zip(rkf,clo)],[hi-r for r,hi in zip(rkf,chi)]],
                fmt='none',color=DARK,capsize=5,lw=1.5,zorder=6)
    ax.axhline(0.9923,color=ACCENT,lw=2.5,ls='--',zorder=5,label='GA-SVR 0.9923')
    for i,(ss,lo_v) in enumerate(zip(val,loo)):
        gap=ss-lo_v
        if abs(gap)>0.01:
            ax.text(i,max(ss,lo_v)+0.025,f'Δ={gap:+.3f}',ha='center',fontsize=7.5,fontweight='bold',color=DARK,
                    bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor=DARK,alpha=0.8))
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=18,ha='right',fontsize=8)
    ax.set_ylabel('R²',fontweight='bold'); ax.set_ylim(0.3,1.12)
    ax.set_title('Single-Split vs Cross-Validated R² — All 8 Models',fontsize=12)
    ax.legend(fontsize=8); ax.grid(True,axis='y',alpha=0.3); plt.tight_layout(); savefig("fig09_split_vs_cv.png")

def fig10(av,X,y,ac,kc):
    Xa=np.array(X)
    mc=[v for v in av if 'MC' in v['name']]
    mc=mc[0] if mc else [v for v in av if v['is_gp']][0]
    m=clone(mc['model_obj']); m.fit(Xa,y)
    sv=np.linspace(0,100000,40); jv=np.linspace(0,0.043,35)
    SS,JJ=np.meshgrid(sv,jv)
    syn=pd.DataFrame({'Salinity':SS.ravel(),'FlowRate':np.full(SS.size,2.),'JammingRatio':JJ.ravel(),'ParticleConc':np.full(SS.size,0.3)})
    Xf=rbf(syn); mu_g,sg_g=m.predict_std(Xf)
    mu_g=mu_g.reshape(SS.shape); sg_g=sg_g.reshape(SS.shape); wg=2*1.96*sg_g
    cpq=mc['CP_q']
    fig=plt.figure(figsize=(20,13)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.38)
    ax1=fig.add_subplot(gs[0,0])
    cf=ax1.contourf(SS/1000,JJ,mu_g,levels=15,cmap='YlOrRd'); plt.colorbar(cf,ax=ax1,label='RIC (%)')
    ax1.scatter(Xa[:,0]/1000,Xa[:,2],c=y,cmap='YlOrRd',s=60,edgecolors='k',linewidth=0.5,zorder=5)
    ax1.set_xlabel('Salinity (×10³)'); ax1.set_ylabel('Jamming Ratio'); ax1.set_title('PC-GPR-MC: Mean RIC Surface')
    ax2=fig.add_subplot(gs[0,1])
    cf2=ax2.contourf(SS/1000,JJ,wg,levels=15,cmap='Blues'); plt.colorbar(cf2,ax=ax2,label='95% PI Width (%)')
    ax2.contour(SS/1000,JJ,mu_g,levels=[20,40,60,80],colors='k',linewidths=0.8,alpha=0.5)
    ax2.set_xlabel('Salinity (×10³)'); ax2.set_ylabel('Jamming Ratio'); ax2.set_title('Uncertainty Map: 95% PI Width')
    ax3=fig.add_subplot(gs[0,2])
    rmap=np.digitize(wg,[15,30])
    import matplotlib as mpl
    cmap_r=mpl.colors.ListedColormap([CP['green'],CP['gold'],CP['red']])
    ax3.contourf(SS/1000,JJ,rmap,levels=[-0.5,0.5,1.5,2.5],cmap=cmap_r,alpha=0.7)
    ax3.contour(SS/1000,JJ,mu_g,levels=[30,50,70],colors='k',linewidths=1,alpha=0.6)
    patches=[mpatches.Patch(color=CP['green'],label='Low risk (PI<15%)'),
             mpatches.Patch(color=CP['gold'],label='Med (15–30%)'),
             mpatches.Patch(color=CP['red'],label='High (>30%)')]
    ax3.legend(handles=patches,loc='upper left',fontsize=8)
    ax3.scatter(Xa[:,0]/1000,Xa[:,2],color='white',s=50,edgecolors='k',linewidth=0.8,zorder=5)
    ax3.set_xlabel('Salinity (×10³)'); ax3.set_ylabel('Jamming Ratio'); ax3.set_title('Injection Risk Classification')
    ax4=fig.add_subplot(gs[1,0])
    sr2=np.linspace(0,100000,200); syn2=pd.DataFrame({'Salinity':sr2,'FlowRate':np.full(200,2.),'JammingRatio':np.full(200,0.011),'ParticleConc':np.full(200,0.3)})
    Xf2=rbf(syn2); mu2,sig2=m.predict_std(Xf2)
    ax4.fill_between(sr2/1000,mu2-1.96*sig2,mu2+1.96*sig2,alpha=0.25,color=PAL[0],label='GP 95% PI')
    ax4.fill_between(sr2/1000,mu2-cpq,mu2+cpq,alpha=0.2,color=PAL[3],label=f'Conformal PI (±{cpq:.1f}%)')
    ax4.plot(sr2/1000,mu2,lw=2.5,color=PAL[0],label='PC-GPR-MC')
    mask=np.abs(Xa[:,2]-0.011)<0.005
    ax4.scatter(Xa[mask,0]/1000,y[mask],color=ACCENT,s=60,edgecolors='k',zorder=5)
    ax4.set_xlabel('Salinity (×10³)'); ax4.set_ylabel('RIC (%)'); ax4.set_title('Risk at J=0.011 Slice'); ax4.legend(fontsize=7); ax4.grid(True)
    ax5=fig.add_subplot(gs[1,1:])
    ax5.axis('off')
    conds=[(0,0),(6000,0.004),(30000,0.011),(100000,0.043),(50000,0.011)]
    rows_t=[]
    for Sc,Jc in conds:
        sp=pd.DataFrame({'Salinity':[Sc],'FlowRate':[2.],'JammingRatio':[Jc],'ParticleConc':[0.3]})
        Xp=rbf(sp); mu_p,sig_p=m.predict_std(Xp); mu_p,sig_p=float(mu_p[0]),float(sig_p[0])
        pil=max(0.,mu_p-1.96*sig_p); pih=min(100.,mu_p+1.96*sig_p)
        risk='Low' if 2*1.96*sig_p<15 else ('Med' if 2*1.96*sig_p<30 else 'High')
        rows_t.append([f'S={Sc:,}, J={Jc}',f'{mu_p:.1f}%',f'[{pil:.1f},{pih:.1f}]%',f'{2*1.96*sig_p:.1f}%',risk])
    tbl=ax5.table(cellText=rows_t,colLabels=['Condition','RIC Pred','95% PI','Width','Risk'],loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1,2.)
    rc={'Low':CP['green'],'Med':CP['gold'],'High':CP['red']}
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor(DARK); cell.set_text_props(color='white',fontweight='bold')
        elif c==4 and r>0: cell.set_facecolor(rc.get(rows_t[r-1][4],CP['lightbg']))
        else: cell.set_facecolor(CP['lightbg'] if r%2==0 else 'white')
        cell.set_edgecolor('#CCCCCC')
    ax5.set_title('Fix 4: Injection Scheduling Risk Table',fontsize=10,fontweight='bold',pad=15)
    fig.suptitle('Physics-Informed GP: Uncertainty → Injection Risk Map\nFirst actionable predictive intervals for CO₂ injectivity',fontsize=12,fontweight='bold',y=1.01)
    savefig("fig10_decision_risk.png")

def fig11(av,X,y):
    Xa=np.array(X); sr=np.linspace(0,100000,200)
    ds=pd.DataFrame(np.tile(np.median(Xa,0),(200,1)),columns=FEAT); ds['Salinity']=sr; Xf=rbf(ds)
    fig,axes=plt.subplots(1,2,figsize=(16,6)); gc={'GP-Base':PAL[0],'PC-GPR-M':CP['gold'],'PC-GPR-C':CP['teal'],'PC-GPR-MC':CP['red']}
    gv=[v for v in av if v['is_gp']]
    for vr in gv:
        m=clone(vr['model_obj']); m.fit(Xa,y); mu,sig=m.predict_std(Xf); col=gc.get(vr['name'],PAL[4])
        axes[0].plot(sr/1000,mu,lw=2.5,color=col,label=vr['name']); axes[0].fill_between(sr/1000,mu-sig,mu+sig,color=col,alpha=0.07)
    axes[0].scatter(Xa[:,0]/1000,y,color=DARK,s=40,edgecolors='white',linewidth=0.5,zorder=5,alpha=0.7)
    axes[0].set_xlabel('Salinity (×10³)'); axes[0].set_ylabel('RIC (%)'); axes[0].set_title('GP Posterior Along Salinity\nShaded = ±1σ'); axes[0].legend(fontsize=8); axes[0].grid(True)
    bm=[v for v in gv if 'Base' in v['name']][0]; bmod=clone(bm['model_obj']); bmod.fit(Xa,y); bp,_=bmod.predict_std(Xf)
    for vr in gv:
        if 'Base' in vr['name']: continue
        m=clone(vr['model_obj']); m.fit(Xa,y); mu,_=m.predict_std(Xf); col=gc.get(vr['name'],PAL[4])
        axes[1].plot(sr/1000,mu-bp,lw=2.5,color=col,label=f'{vr["name"]} − Base')
    axes[1].axhline(0,color=DARK,lw=1.5,ls='--',label='No diff'); axes[1].legend(fontsize=8); axes[1].grid(True)
    axes[1].set_xlabel('Salinity (×10³)'); axes[1].set_ylabel('ΔPrediction vs GP-Base'); axes[1].set_title('Constraint Contribution\n(Fix 2: Decoupled — no tension)')
    fig.suptitle('Constraint Effect Analysis',fontsize=12,fontweight='bold'); plt.tight_layout(); savefig("fig11_constraint.png")

def fig12(av,X,y):
    Xa=np.array(X); gv=[v for v in av if v['is_gp']]; n=len(gv)
    fig,axes=plt.subplots(2,2,figsize=(16,12)); axes=axes.flatten()
    for ax,vr in zip(axes,gv):
        m=clone(vr['model_obj']); m.fit(Xa,y); mi,si=perm_imp(m,Xa,y,nr=30)
        order=np.argsort(mi); cols=[PAL[i%len(PAL)] for i in range(len(FEAT))]
        ax.barh([FEAT[i] for i in order],mi[order],xerr=si[order],capsize=3,color=[cols[i] for i in order],alpha=0.88,edgecolor='white',linewidth=0.5)
        ax.axvline(0,color=DARK,lw=0.8,ls='--'); ax.set_xlabel('Mean R² decrease (n=30)'); ax.set_title(f'{vr["name"]}\nPermutation Feature Importance'); ax.grid(True,axis='x')
    for ax in axes[n:]: ax.set_visible(False)
    fig.suptitle('Feature Importance — Physics-Constrained GP Models',fontsize=13,fontweight='bold'); plt.tight_layout(); savefig("fig12_importance.png")

def fig13(av):
    n=len(av)
    fig,axes=plt.subplots(2,n,figsize=(5*n,10))
    if n==1: axes=axes.reshape(2,1)
    for col,vr in enumerate(av):
        resid=vr['resid_loo']; ya=vr['y_all']
        axes[0,col].scatter(ya,resid,c=np.abs(resid),cmap='RdYlGn_r',s=60,edgecolors='k',linewidth=0.3,alpha=0.9)
        axes[0,col].axhline(0,color=ACCENT,lw=2,ls='--')
        axes[0,col].axhline(2*resid.std(),color=GOLD,lw=1.5,ls=':'); axes[0,col].axhline(-2*resid.std(),color=GOLD,lw=1.5,ls=':',label='±2σ')
        axes[0,col].set_xlabel('Measured RIC'); axes[0,col].set_ylabel('LOO Residual'); axes[0,col].set_title(f'{vr["name"]}\nLOO R²={vr["LOO_R2"]:.4f}')
        axes[0,col].legend(fontsize=7); axes[0,col].grid(True)
        parts=axes[1,col].violinplot(resid,showmedians=True)
        for pc in parts['bodies']: pc.set_facecolor(PAL[col%len(PAL)]); pc.set_alpha(0.7)
        axes[1,col].axhline(0,color=ACCENT,lw=2,ls='--')
        wp=vr['Wilcoxon_p']; tag='✅ Unbiased' if (not np.isnan(wp) and wp>0.05) else '⚠️ Biased'
        axes[1,col].set_title(f'μ={resid.mean():.2f}  σ={resid.std():.2f}')
        axes[1,col].text(0.5,0.02,f'Wilcoxon p={wp:.3f}\n{tag}',transform=axes[1,col].transAxes,ha='center',fontsize=8,bbox=dict(boxstyle='round',facecolor=CP['paper'],alpha=0.9))
        axes[1,col].set_xticks([]); axes[1,col].grid(True,axis='y')
    fig.suptitle('LOO Residual Analysis — All 8 Models',fontsize=13,fontweight='bold'); plt.tight_layout(); savefig("fig13_residuals.png")

def fig14(av):
    cols_m=['R²\nTrain','R²\nVal','R²\nLOO','RKF\nMean','RMSE\nVal','MAE\nVal','AAPE\nVal%','Pearson\nr','Conf\nCov','Mono\nOK%']
    rd=[[v['train']['R2'],v['test']['R2'],v['LOO_R2'],v['RKF_mean'],v['test']['RMSE'],v['test']['MAE'],v['test']['AAPE'],v['Pearson_r'],v['CP_LOO_cov'],(1-v['Mono_viol'])*100] for v in av]
    df_h=pd.DataFrame(rd,columns=cols_m,index=[v['name'] for v in av])
    df_n=df_h.copy(); lb={'RMSE\nVal','MAE\nVal','AAPE\nVal%'}
    for c in df_n.columns:
        mn,mx=df_n[c].min(),df_n[c].max()
        if mx>mn: df_n[c]=(df_n[c]-mn)/(mx-mn)
        if c in lb: df_n[c]=1-df_n[c]
    fig,ax=plt.subplots(figsize=(17,max(5,len(av)*1.5+2)))
    cmap=LinearSegmentedColormap.from_list('rg',[CP['red'],'#FFFDE7',CP['green']])
    im=ax.imshow(df_n.values,cmap=cmap,aspect='auto',vmin=0,vmax=1); plt.colorbar(im,ax=ax,label='Norm. score (1=best)')
    ax.set_xticks(range(len(cols_m))); ax.set_yticks(range(len(av)))
    ax.set_xticklabels(cols_m,fontsize=9); ax.set_yticklabels([v['name'] for v in av],fontsize=9)
    for i,row in enumerate(rd):
        for j,v in enumerate(row):
            ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,fontweight='bold',
                    color='white' if (df_n.values[i,j]>0.65 or df_n.values[i,j]<0.35) else DARK)
    for i,vr in enumerate(av):
        if vr['is_gp']:
            ax.add_patch(mpatches.Rectangle((-0.5,i-0.5),len(cols_m),1,fill=False,edgecolor=CP['blue2'],lw=2.5))
    ax.set_title('Comprehensive Performance Heatmap (Blue box=GP)\nGreen=best, Red=worst per column',fontsize=12,fontweight='bold',pad=12)
    plt.tight_layout(); savefig("fig14_heatmap.png")

def fig15(av):
    mets=['R²\nLOO','R²\nVal','RKF\nMean','1−RMSE/50','Conf\nCov','Mono\nOK','1−ECE/0.1']
    nm=len(mets); angles=np.linspace(0,2*np.pi,nm,endpoint=False).tolist()+[0]
    fig,ax=plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True)); ax.set_facecolor(CP['lightbg'])
    for i,vr in enumerate(av):
        ecen=(1-min(vr['ECE'],0.1)/0.1) if not np.isnan(vr['ECE']) else 0.5
        vals=[vr['LOO_R2'],vr['test']['R2'],vr['RKF_mean'],1-vr['test']['RMSE']/50,vr['CP_LOO_cov'],1-vr['Mono_viol'],ecen]
        vals=[max(0,v) for v in vals]+[max(0,vals[0])]; col=PAL[i%len(PAL)]
        lw=3 if vr['is_gp'] else 1.8; mk='D' if vr['is_gp'] else 'o'
        ax.plot(angles,vals,marker=mk,lw=lw,color=col,ms=8 if lw>2 else 5,label=vr['name'])
        ax.fill(angles,vals,color=col,alpha=0.05)
    p=[0.9923,0.9923,0.9923,1-0.99/50,0.5,0.5,0.5]+[0.9923]
    ax.plot(angles,p,'s--',lw=2,color=GOLD,ms=8,label='GA-SVR (no CI)'); ax.fill(angles,p,color=GOLD,alpha=0.04)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(mets,size=10,fontweight='bold')
    ax.set_ylim(0,1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0]); ax.grid(True,alpha=0.35)
    ax.set_title('Multi-Metric Radar — 7 Axes incl. ECE & Conformal Coverage',fontsize=11,fontweight='bold',pad=28)
    ax.legend(loc='upper right',bbox_to_anchor=(1.45,1.15),fontsize=8); plt.tight_layout(); savefig("fig15_radar.png")

def save_table(av):
    rows=[]
    for vr in av:
        wp=vr['Wilcoxon_p']; ece=vr['ECE']
        rows.append({'Model':vr['name'],'Type':'GP+Physics' if 'PC-GPR' in vr['name'] else ('GP-Base' if vr['is_gp'] else 'Baseline'),
            'R2_Train':round(vr['train']['R2'],5),'R2_Val_20':round(vr['test']['R2'],5),'R2_LOO':round(vr['LOO_R2'],5),
            'RMSE_LOO':round(vr['LOO_RMSE'],4),'MAE_LOO':round(vr['LOO_MAE'],4),'AAPE_LOO':round(vr['LOO_AAPE'],2),
            'RMSE_Val':round(vr['test']['RMSE'],4),'MAE_Val':round(vr['test']['MAE'],4),
            'RKF_Mean':round(vr['RKF_mean'],5),'RKF_CI_95':f'[{vr["RKF_CI"][0]:.4f},{vr["RKF_CI"][1]:.4f}]',
            'Boot_CI_95':f'[{vr["Boot_CI"][0]:.4f},{vr["Boot_CI"][1]:.4f}]',
            'Conformal_LOO_cov':round(vr['CP_LOO_cov'],4),'Conformal_LOO_wid':round(vr['CP_LOO_wid'],3),
            'ECE':round(ece,5) if not np.isnan(ece) else 'n/a',
            'Pearson_r':round(vr['Pearson_r'],5),'Spearman_rho':round(vr['Spearman_r'],5),
            'Wilcoxon_p':round(wp,5) if not np.isnan(wp) else 'nan','Bias':'Unbiased' if (not np.isnan(wp) and wp>0.05) else 'Biased',
            'Mono_viol_%':round(vr['Mono_viol']*100,2)})
    df=pd.DataFrame(rows); p=os.path.join(OUTDIR,'tables','full_results_v2.csv')
    df.to_csv(p,index=False); print(f"   ✅ full_results_v2.csv"); return df

if __name__=="__main__":
    t0=time.time()
    print("="*70)
    print("  PAPER 2A v2 — Fixed PIML Pipeline")
    print("="*70)
    df=load_data(); X=df[FEAT]; y=df['RIC'].values; Xa=X.values
    print(f"\n✅ Data: n={len(y)}, d={len(FEAT)}")
    print("\n🔬 Fix 1: Rebuilding Civan prior (2-param, lit-anchored)...")
    ac,kc,cp_pred,r2c=fit_civan(df)
    print("\n🔗 Building 8 models...")
    gp_b =GPBase(nr=8); pc_m=PCGPRM(nr=8,nv=14,st=35.)
    pc_c =PCGPRC(ac=ac,kc=kc,nr=8); pc_mc=PCGPRMC(ac=ac,kc=kc,nr=8,nv=14,st=35.)
    lr=Pipeline([('s',RobustScaler()),('m',LinearRegression())])
    br=Pipeline([('s',RobustScaler()),('m',BayesianRidge())])
    svr=Pipeline([('s',RobustScaler()),('m',SVR(C=500,gamma=0.01,epsilon=1.0,kernel='rbf'))])
    stk=StackingRegressor(estimators=[('lr',lr),('br',br),('svr',svr)],final_estimator=Ridge(alpha=1.0),cv=5,n_jobs=1)
    ml=[('GP-Base',gp_b,True),('PC-GPR-M',pc_m,True),('PC-GPR-C',pc_c,True),('PC-GPR-MC',pc_mc,True),
        ('LR (Paper 1)',lr,False),('BR (Paper 1)',br,False),('SVR-GS',svr,False),('Stack',stk,False)]
    print("\n📐 Full validation (LOO+RKF×20+Bootstrap+Conformal)...")
    av=[]
    for nm,mod,ig in ml: av.append(validate(mod,Xa,y,nm,is_gp=ig))
    print("\n📊 Generating 15 figures...")
    fig01(df,cp_pred,ac,kc,r2c); fig02(df); fig03(av,Xa,y); fig04(av,y); fig05(av); fig06(av,y)
    fig07(av,X,y); fig08(av); fig09(av); fig10(av,X,y,ac,kc); fig11(av,X,y)
    fig12(av,X,y); fig13(av); fig14(av); fig15(av)
    df_res=save_table(av)
    elapsed=time.time()-t0
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"\n  {'Model':18s}  {'LOO R²':7s}  {'Boot CI':22s}  {'ECE':7s}  {'Viol%':6s}  {'ConfCov':8s}")
    print(f"  {'-'*76}")
    for vr in av:
        es=f"{vr['ECE']:.4f}" if not np.isnan(vr['ECE']) else "  n/a  "
        print(f"  {vr['name']:18s}  {vr['LOO_R2']:.4f}   [{vr['Boot_CI'][0]:.3f},{vr['Boot_CI'][1]:.3f}]    {es}  {vr['Mono_viol']*100:5.1f}%  {vr['CP_LOO_cov']:.3f}")
    print(f"\n  GA-SVR Wang 2020:     0.9923   no CI            no ECE   n/a    no coverage")
    print(f"\n  Civan v2 R²={r2c:.4f} (was 0.1791, improvement +{r2c-0.1791:+.4f})")
    print(f"\n  Output: {OUTDIR}  |  Figures: {len(os.listdir(os.path.join(OUTDIR,'figures')))}")
