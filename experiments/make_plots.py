"""Report-ready figures from saved numerical results; no model selection here."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.baselines import prepare
from harness import dayahead as h
OUT=ROOT/'results/plots'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})


def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    df,truth,baselines=prepare()
    table=pd.read_csv(ROOT/'results/stage2_period_metrics.csv')
    monthly=table[table.frequency=='M'].copy()
    monthly['date']=pd.to_datetime(monthly.period+'-01')
    for task,metric,label,families in [('demand','MAPE','Demand MAPE (%)',['ridge','boost_demand']),
                                      ('price','Acc','Price accuracy (%)',['logistic','boost_price'])]:
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        for ax,family in zip(axes,families):
            sub=monthly[(monthly.task==task)&(monthly.model==family)&(monthly.strategy=='frozen')]
            ax.plot(sub.date,sub[metric],color='#164c82',lw=1,label=family)
            ax.fill_between(sub.date,sub.day_low,sub.day_high,color='#72a7cf',alpha=.35,label='95% whole-day bootstrap')
            ax.fill_between(sub.date,sub.iid_low,sub.iid_high,color='#164c82',alpha=.3,label='95% independent half-hours')
            for baseline in ('baseline_1d','baseline_7d'):
                b=monthly[(monthly.task==task)&(monthly.model==baseline)]
                ax.plot(b.date,b[metric],lw=.7,alpha=.7,label=baseline)
            ax.set_ylabel(label)
            ax.legend(ncol=2,fontsize=8,loc='best')
        axes[-1].set_xlabel('Market month (AEST)')
        save(fig,f'frozen_{task}_confidence_intervals')
        fig,ax=plt.subplots(figsize=(11,3.7))
        chosen='boost_demand' if task=='demand' else 'boost_price'
        for strategy in ('frozen','periodic_3y','input_drift_3y','error_drift_3y'):
            sub=monthly[(monthly.task==task)&(monthly.model==chosen)&(monthly.strategy==strategy)]
            ax.plot(sub.date,sub[metric],label=strategy,lw=1)
        for baseline in ('baseline_1d','baseline_7d'):
            sub=monthly[(monthly.task==task)&(monthly.model==baseline)]
            ax.plot(sub.date,sub[metric],label=baseline,lw=.7,alpha=.6)
        ax.set(xlabel='Market month (AEST)',ylabel=label)
        ax.legend(ncol=3,fontsize=8)
        save(fig,f'adaptation_{task}')
        # Adapted confidence intervals shown separately for legibility.
        fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
        for ax,strategy in zip(axes,('periodic_3y','input_drift_3y','error_drift_3y')):
            sub=monthly[(monthly.task==task)&(monthly.model==chosen)&(monthly.strategy==strategy)]
            ax.plot(sub.date,sub[metric],label=strategy,lw=.8)
            ax.fill_between(sub.date,sub.day_low,sub.day_high,alpha=.25,label='95% day bootstrap')
            ax.fill_between(sub.date,sub.iid_low,sub.iid_high,alpha=.35,label='95% independent')
            ax.set_ylabel(label)
            ax.legend(ncol=3,fontsize=7)
        axes[-1].set_xlabel('Market month (AEST)')
        save(fig,f'adapted_{task}_confidence_intervals')
    profiles=pd.read_csv(ROOT/'results/cluster_profiles.csv',index_col=0)
    shares=pd.read_csv(ROOT/'results/cluster_yearly_shares.csv',index_col=0)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for cluster,row in profiles.iterrows():
        axes[0].plot(np.arange(48)/2,row,label=f'Type {cluster}')
    axes[0].set(xlabel='Interval start hour (AEST)',ylabel='Demand / daily mean (ratio)')
    axes[0].legend()
    for cluster in shares.columns:
        axes[1].plot(shares.index,shares[cluster],marker='.',label=f'Type {cluster}')
    axes[1].axvline(2013.5,color='grey',ls='--',lw=.7)
    axes[1].set(xlabel='Market year',ylabel='Days assigned to type (%)')
    axes[1].legend()
    save(fig,'day_types')
    relation=pd.read_csv(ROOT/'results/day_type_error_relation.csv',parse_dates=['day']).set_index('day')
    rel=relation.resample('MS').agg({'unusual':'mean','demand_loss':'mean','price_loss':'mean','input_statistic':'mean'})
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
    axes[0].plot(rel.index,100*rel.unusual,label='Unusual demand profiles')
    axes[0].set_ylabel('Unusual days (%)')
    axes[1].plot(rel.index,rel.demand_loss,label='Frozen demand MAPE')
    axes[1].set_ylabel('Demand MAPE (%)')
    axes[2].plot(rel.index,rel.input_statistic,label='Input drift statistic')
    config=json.loads((ROOT/'results/detector_config_demand.json').read_text())
    axes[2].axhline(config['input_threshold'],ls='--',color='red',label='Threshold calibrated in 2014')
    axes[2].set(ylabel='Mean absolute z-score',xlabel='Market month (AEST)')
    for ax in axes: ax.legend(fontsize=8)
    save(fig,'day_types_drift_error')
    signals=pd.read_csv(ROOT/'results/drift_signals.csv',parse_dates=['day'])
    fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True)
    for ax,task,metric in zip(axes,('demand','price'),('MAPE','Acc')):
        chosen='boost_demand' if task=='demand' else 'boost_price'
        sub=monthly[(monthly.task==task)&(monthly.model==chosen)&(monthly.strategy=='frozen')]
        ax.plot(sub.date,sub[metric],color='black',lw=1,label='Frozen model')
        for detector,color in [('input','#b06121'),('error','#155b9c')]:
            ss=signals[(signals.task==task)&(signals.strategy=='frozen')&(signals.detector==detector)]
            for j,day in enumerate(ss.day): ax.axvline(day,color=color,alpha=.45,lw=.7,label=detector+' alarm' if j==0 else None)
        ax.set_ylabel('MAPE (%)' if task=='demand' else 'Accuracy (%)')
        ax.legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel('Market month (AEST)')
    save(fig,'drift_alarm_dates')
    daily=pd.read_csv(ROOT/'results/daily_class_balance.csv',parse_dates=['day']).set_index('day')
    fig,axes=plt.subplots(2,1,figsize=(11,5))
    share=daily.positive_pct.resample('MS').mean()
    axes[0].plot(share.index,share)
    axes[0].set(xlabel='Market month (AEST)',ylabel='Positive target class (%)')
    axes[1].scatter(daily.previous_mean_price,daily.positive_pct,s=3,alpha=.2)
    axes[1].set_xscale('symlog',linthresh=100)
    axes[1].set(xlabel='Previous day mean NSW price ($/MWh; symlog scale)',ylabel='Positive target class (%)')
    save(fig,'class_balance')
    # Two fixed illustrative weeks, not selected by largest/smallest errors.
    for first in ('2016-07-04','2026-07-06'):
        last=pd.Timestamp(first)+pd.Timedelta(days=7)
        days=h.market_day(df.index)
        keep=(days>=first)&(days<last)
        idx=df.index[keep]
        fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True)
        axes[0].plot(idx,truth.loc[idx,'nsw_demand_mw'],label='Observed',lw=1)
        axes[0].plot(idx,baselines['baseline_1d'].loc[idx,'nsw_demand_mw'],label='Yesterday baseline',lw=.7,alpha=.7)
        p=pd.read_parquet(ROOT/'results/demand_periodic_3y_predictions.parquet')
        axes[0].plot(idx,p.loc[idx,'nsw_demand_mw'],label='Scheduled retraining',lw=.9)
        axes[0].set_ylabel('NSW demand (MW)')
        p=pd.read_parquet(ROOT/'results/price_input_drift_3y_predictions.parquet')
        axes[1].plot(idx,truth.loc[idx,'nsw_price_up'],label='Observed class',lw=.9)
        axes[1].plot(idx,p.loc[idx,'nsw_price_up']+.04,label='Input-drift model (+0.04 for visibility)',lw=.6,alpha=.8)
        axes[1].set(ylabel='Price-direction class (0/1)',xlabel='Interval end timestamp (AEST)')
        for ax in axes: ax.legend(fontsize=7,ncol=3)
        save(fig,'prediction_week_'+first)
    # Numeric CI comparison supports interpretation rather than overlap-based testing.
    widths=monthly.assign(iid_width=monthly.iid_high-monthly.iid_low,day_width=monthly.day_high-monthly.day_low)
    widths.groupby(['task','model','strategy'])[['iid_width','day_width']].mean().to_csv(ROOT/'results/ci_width_summary.csv')
    # Cluster interpretation evidence: weekdays, seasons, demand level and drift.
    dt=pd.read_csv(ROOT/'results/day_types.csv',parse_dates=['day']).set_index('day')
    dt['weekend']=dt.index.dayofweek>=5
    dt['month']=dt.index.month
    dt['mean_demand']=daily.demand_mean
    dt['training_period']=dt.index<'2014-01-01'
    dt.groupby(['training_period','cluster']).agg(days=('month','size'),weekend_pct=('weekend',lambda s:100*s.mean()),
                  mean_demand_mw=('mean_demand','mean'),unusual_pct=('unusual',lambda s:100*s.mean())).to_csv(ROOT/'results/cluster_interpretation.csv')
    print(f'Created figures in {OUT}',flush=True)

if __name__=='__main__': main()
