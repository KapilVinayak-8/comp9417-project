"""Simple Stage 2: fixed day types, online drift, and budgeted retraining.
Run: .venv\Scripts\python.exe experiments\stage2.py
All choices below are fixed before reading the 2015+ simulation outcomes.
"""
from pathlib import Path
import sys
import json
from collections import deque
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forecast.model import features
from experiments.baselines import prepare
from experiments.stage1 import make_model
from harness import dayahead as h
SEED=9417
WINDOW_DAYS=3*365
BUDGET=6
COOLDOWN_DAYS=540
PERSISTENCE=7
MONITOR_DAYS=28


def confidence_intervals(values, seed=SEED):
    """Percent-valued half-hour errors/correct indicators, complete equal-length days.
    1000 whole-day resamples are equivalent to resampling equal-weight daily means.
    """
    a=np.asarray(values,float)
    assert len(a)%48==0
    mean=a.mean()
    half=1.96*a.std(ddof=1)/np.sqrt(len(a))
    daily=a.reshape(-1,48).mean(axis=1)
    rng=np.random.default_rng(seed)
    boots=daily[rng.integers(0,len(daily),size=(1000,len(daily)))].mean(axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    return dict(iid_low=float(mean-half),iid_high=float(mean+half),
                day_low=float(lo),day_high=float(hi))


def period_metrics(index,y,p,task,name,strategy):
    days=h.market_day(index)
    rows=[]
    for frequency in ('Y','M'):
        labels=days.to_period(frequency)
        for period in labels.unique():
            mask=labels==period
            true=np.asarray(y)[mask]
            pred=np.asarray(p)[mask]
            if task=='demand':
                errors=100*np.abs(true-pred)/true
                metrics={'MAPE':errors.mean(),'MAE':np.abs(true-pred).mean()}
            else:
                errors=100*(true==pred).astype(float)
                metrics={'Acc':errors.mean(),'BalAcc':h.balanced_accuracy(true,pred)}
            row={'model':name,'strategy':strategy,'task':task,'frequency':frequency,
                 'period':str(period),'n':len(true),**metrics}
            if frequency=='M':
                row.update(confidence_intervals(errors))
                if task=='price':
                    # Appendix A binomial normal approximation, in percentage points.
                    radius=1.96*np.sqrt(metrics['Acc']*(100-metrics['Acc'])/len(true))
                    row['iid_low']=metrics['Acc']-radius
                    row['iid_high']=metrics['Acc']+radius
                if task=='price':
                    # Normal binomial interval above; also retain Hoeffding bound.
                    radius=100*np.sqrt(np.log(40)/(2*len(true)))
                    row.update(hoeffding_low=max(0,errors.mean()-radius),
                               hoeffding_high=min(100,errors.mean()+radius))
                    for key in ('iid_low','day_low'): row[key]=max(0,row[key])
                    for key in ('iid_high','day_high'): row[key]=min(100,row[key])
            rows.append(row)
    return rows


def day_types(df):
    days=h.market_day(df.index)[::48]
    raw=df.nsw_demand.to_numpy().reshape(-1,48)
    shape=raw/raw.mean(axis=1,keepdims=True)
    train=days<'2014-01-01'
    pca=PCA(n_components=.95,svd_solver='full').fit(shape[train])
    z=pca.transform(shape)
    scores=[]
    models={}
    for k in range(2,7):
        km=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(z[train])
        score=silhouette_score(z[train],km.labels_,sample_size=min(800,train.sum()),random_state=SEED)
        scores.append({'k':k,'silhouette':score})
        models[k]=km
    best=max(scores,key=lambda r:r['silhouette'])['k']
    km=models[best]
    labels=km.predict(z)
    distance=km.transform(z).min(axis=1)
    cutoff=np.quantile(distance[train],.99)
    daily=pd.DataFrame({'cluster':labels,'distance':distance,'unusual':distance>cutoff},index=days)
    profiles=pd.DataFrame([shape[train][km.labels_==i].mean(axis=0) for i in range(best)])
    profiles.index.name='cluster'
    profiles.to_csv(ROOT/'results/cluster_profiles.csv')
    shares=pd.crosstab(days.year,labels,normalize='index')*100
    shares.index.name='year'
    shares.to_csv(ROOT/'results/cluster_yearly_shares.csv')
    daily.to_csv(ROOT/'results/day_types.csv',index_label='day')
    pd.DataFrame(scores).to_csv(ROOT/'results/cluster_selection.csv',index=False)
    metadata={'normalisation':'divide each daily demand profile by its mean; shape only',
              'fit_period':'2010-2013 only; never refitted','pca_components':int(pca.n_components_),
              'explained_variance':float(pca.explained_variance_ratio_.sum()),
              'selected_k':best,'unusual_distance_cutoff_training_99pct':float(cutoff)}
    (ROOT/'results/cluster_config.json').write_text(json.dumps(metadata,indent=2))
    joblib.dump({'pca':pca,'kmeans':km},ROOT/'results/day_type_models.joblib')
    print('Day types:',metadata,flush=True)
    return daily


def input_statistics(df):
    """Monitors only observed inputs, at end of day D. No forecast errors/targets."""
    days=h.market_day(df.index)
    daily=pd.DataFrame(index=days.unique())
    daily['demand_mean']=df.nsw_demand.groupby(days).mean()
    daily['demand_range']=df.nsw_demand.groupby(days).max()-df.nsw_demand.groupby(days).min()
    daily['price_asinh_mean']=np.arcsinh(df.nsw_price/100).groupby(days).mean()
    daily['temperature']=df.syd_temp.groupby(days).mean()
    daily['solar']=df.syd_solar.groupby(days).mean()
    ref=daily.loc['2010':'2013']
    means=ref.groupby(ref.index.month).mean()
    scales=ref.groupby(ref.index.month).std().clip(lower=1e-6)
    # Monthly references reduce predictable seasonal signals without using later data.
    z=(daily.to_numpy()-means.loc[daily.index.month].to_numpy())/scales.loc[daily.index.month].to_numpy()
    point=pd.Series(np.abs(z).mean(axis=1),index=daily.index)
    monitor=point.rolling(MONITOR_DAYS).mean()
    threshold=float(monitor.loc['2014'].quantile(.99))
    return point,monitor,threshold


def error_reference(loss,index):
    daily=pd.Series(loss,index=h.market_day(index)).groupby(level=0).mean()
    means=daily.groupby(daily.index.month).mean()
    scale=max(float((daily-daily.index.month.map(means)).std()),1e-6)
    standardized=(daily-daily.index.month.map(means))/scale
    threshold=float(standardized.rolling(MONITOR_DAYS).mean().dropna().quantile(.99))
    return means,scale,threshold


class Detector:
    """One-sided rolling mean, seven confirmations, 540-day cooldown.
    A detector sees day D only after its forecast has been evaluated.
    """
    def __init__(self,threshold):
        self.threshold=threshold
        self.buffer=deque(maxlen=MONITOR_DAYS)
        self.streak=0
        self.last=None
    def observe(self,day,value):
        self.buffer.append(float(value))
        statistic=float(np.mean(self.buffer)) if len(self.buffer)==MONITOR_DAYS else np.nan
        above=bool(np.isfinite(statistic) and statistic>self.threshold)
        self.streak=self.streak+1 if above else 0
        eligible=self.last is None or (day-self.last).days>=COOLDOWN_DAYS
        alarm=bool(eligible and self.streak>=PERSISTENCE)
        if alarm:
            self.last=day
            self.streak=0
        return statistic,alarm
    def reset_errors(self):
        self.buffer.clear()
        self.streak=0


def main():
    out=ROOT/'results'
    df,truth,baseline=prepare()
    x=features(df)
    days=h.market_day(df.index)
    daily_types=day_types(df)
    point,input_monitor,input_threshold=input_statistics(df)
    config=json.loads((out/'stage1_frozen/config.json').read_text())
    valid=x.notna().all(axis=1).to_numpy()
    train=valid & (days<'2014-01-01')
    validation=valid & (days>='2014-01-01') & (days<'2015-01-01')
    sim=days>='2015-01-01'
    dates=days[sim][::48]
    simulation_start=int(np.flatnonzero(sim)[0])
    all_metrics=[]
    summary=[]
    alarms=[]
    retrains=[]
    # Frozen histories of every model studied in Stage 1, not only selected models.
    for family,task in [('ridge','demand'),('boost_demand','demand'),('logistic','price'),('boost_price','price')]:
        target='nsw_demand_mw' if task=='demand' else 'nsw_price_up'
        prediction=pd.read_parquet(out/f'stage1_frozen/{family}_simulation.parquet')[target]
        all_metrics.extend(period_metrics(prediction.index,truth.loc[prediction.index,target],prediction,task,family,'frozen'))
    for name,pred in baseline.items():
        for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
            all_metrics.extend(period_metrics(df.index[sim],truth.loc[sim,col],pred.loc[sim,col],task,name,'baseline'))
    schedules={pd.Timestamp(f'{year}-01-01') for year in (2016,2018,2020,2022,2024,2026)}
    for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
        chosen=config['selected'][task]
        family=chosen['family']
        setting=chosen['setting']
        calibrator=make_model(family,setting)
        calibrator.fit(x.loc[train],truth.loc[train,col])
        cp=calibrator.predict(x.loc[validation])
        cy=truth.loc[validation,col].to_numpy()
        calibration_loss=100*np.abs(cy-cp)/cy if task=='demand' else 100*(cy!=cp)
        emean,escale,ethreshold=error_reference(calibration_loss,df.index[validation])
        (out/f'detector_config_{task}.json').write_text(json.dumps({
            'input_reference':'2010-2013 monthly means and standard deviations',
            'input_variables':['daily mean demand','daily demand range','daily mean asinh(price/100)','daily mean Sydney temperature','daily mean Sydney solar'],
            'input_threshold':input_threshold,'error_threshold':ethreshold,
            'error_reference':'out-of-sample 2014 daily losses; calendar-month centering',
            'error_month_means':{str(k):float(v) for k,v in emean.items()},'error_scale':escale,
            'threshold_calibration':'99th percentile of available 2014 trailing 28-day statistics',
            'rolling_days':MONITOR_DAYS,'consecutive_days':PERSISTENCE,'cooldown_days':COOLDOWN_DAYS,
            'retraining_budget':BUDGET,'training_window_days':WINDOW_DAYS},indent=2))
        initial=joblib.load(out/f'stage1_frozen/{family}.joblib')
        for strategy in ('frozen','periodic_3y','input_drift_3y','error_drift_3y'):
            model=initial
            input_detector=Detector(input_threshold)
            error_detector=Detector(ethreshold)
            # Input monitoring may retain predeployment observations. Error monitoring
            # is reset at deployment because frozen parameters changed after refitting.
            for d in point.loc['2014'].index[-MONITOR_DAYS:]:
                input_detector.observe(d,point.loc[d])
            predictions=np.empty(sim.sum())
            count=0
            for i,day in enumerate(dates):
                # A periodic replacement was decided in advance; training stops D-1.
                scheduled=strategy=='periodic_3y' and day in schedules
                if scheduled and count<BUDGET:
                    cutoff=day
                    fitmask=valid & (days<cutoff) & (days>=cutoff-pd.Timedelta(days=WINDOW_DAYS))
                    model=make_model(family,setting)
                    model.fit(x.loc[fitmask],truth.loc[fitmask,col])
                    count+=1
                    retrains.append({'task':task,'strategy':strategy,'decision_day':str((day-pd.Timedelta(days=1)).date()),
                                     'effective_day':str(day.date()),'number':count,'train_end':str(days[fitmask].max().date()),'rows':int(fitmask.sum())})
                # Batch precomputed causal features are equivalent to make_inputs(day).
                idx=slice(i*48,(i+1)*48)
                absolute=simulation_start+i*48
                xday=x.iloc[absolute:absolute+48]
                pred=model.predict(xday)
                predictions[idx]=pred
                y=truth[col].iloc[absolute:absolute+48].to_numpy()
                loss=float(np.mean(100*np.abs(y-pred)/y)) if task=='demand' else float(100*np.mean(y!=pred))
                istat,ialarm=input_detector.observe(day,point.loc[day])
                estat,ealarm=error_detector.observe(day,(loss-emean.loc[day.month])/escale)
                for detector,stat,alarm,threshold in [('input',istat,ialarm,input_threshold),('error',estat,ealarm,ethreshold)]:
                    if alarm:
                        alarms.append({'task':task,'strategy':strategy,'detector':detector,'day':str(day.date()),
                                       'statistic':stat,'threshold':threshold,'daily_loss':loss})
                trigger=(strategy=='input_drift_3y' and ialarm) or (strategy=='error_drift_3y' and ealarm)
                if trigger and count<BUDGET and i<len(dates)-1:
                    # Learn only after day D ends; replacement first predicts D+1.
                    cutoff=day+pd.Timedelta(days=1)
                    fitmask=valid & (days<cutoff) & (days>=cutoff-pd.Timedelta(days=WINDOW_DAYS))
                    model=make_model(family,setting)
                    model.fit(x.loc[fitmask],truth.loc[fitmask,col])
                    count+=1
                    retrains.append({'task':task,'strategy':strategy,'decision_day':str(day.date()),
                                     'effective_day':str(cutoff.date()),'number':count,'train_end':str(days[fitmask].max().date()),'rows':int(fitmask.sum())})
                    error_detector.reset_errors()
            assert count<=BUDGET
            pd.DataFrame({col:predictions},index=df.index[sim]).to_parquet(out/f'{task}_{strategy}_predictions.parquet')
            if strategy!='frozen':
                all_metrics.extend(period_metrics(df.index[sim],truth.loc[sim,col],predictions,task,family,strategy))
            for split,start,end in [('development','2015-01-01','2025-01-01'),('holdout','2025-01-01','2026-11-01'),('all','2015-01-01','2026-11-01')]:
                keep=(days[sim]>=start)&(days[sim]<end)
                y=truth.loc[sim,col].to_numpy()[keep]
                p=predictions[keep]
                metrics=({'MAPE':h.mape(y,p),'MAE':h.mae(y,p)} if task=='demand'
                         else {'Acc':h.accuracy(y,p),'BalAcc':h.balanced_accuracy(y,p)})
                used=sum(r['task']==task and r['strategy']==strategy and pd.Timestamp(r['effective_day'])<pd.Timestamp(end) for r in retrains)
                summary.append({'task':task,'strategy':strategy,'split':split,'retrains_to_end':used,**metrics})
            print(task,strategy,summary[-1],flush=True)
    pd.DataFrame(all_metrics).to_csv(out/'stage2_period_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'adaptation_summary.csv',index=False)
    pd.DataFrame(alarms).to_csv(out/'drift_signals.csv',index=False)
    pd.DataFrame(retrains).to_csv(out/'retraining_log.csv',index=False)
    # Selection ONLY uses 2015-2024, then reports holdout without revising policy.
    final={}
    table=pd.DataFrame(summary)
    for task in ('demand','price'):
        subset=table[(table.task==task)&(table.split=='development')]
        best=subset.sort_values('MAPE' if task=='demand' else 'BalAcc',ascending=task=='demand').iloc[0]
        final[task]={'strategy':best.strategy,**config['selected'][task]}
    (out/'final_strategy.json').write_text(json.dumps({'selection_period':'2015-2024 only','holdout':'2025-October 2026, no policy revision',
        'window_days':WINDOW_DAYS,'selected':final},indent=2))
    # Relate fixed-reference day types and input monitoring to frozen daily errors.
    relation=daily_types.copy()
    relation['input_statistic']=input_monitor
    relation['input_above_threshold']=input_monitor>input_threshold
    for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
        p=pd.read_parquet(out/f'{task}_frozen_predictions.parquet')[col]
        y=truth.loc[p.index,col]
        error=100*np.abs(y-p)/y if task=='demand' else 100*(y!=p)
        relation[task+'_loss']=error.groupby(h.market_day(p.index)).mean()
    relation.loc['2015':].to_csv(out/'day_type_error_relation.csv',index_label='day')
    monthly=relation.loc['2015':].groupby(relation.loc['2015':].index.to_period('M')).agg(
        unusual_share=('unusual','mean'),input_statistic=('input_statistic','mean'),
        demand_loss=('demand_loss','mean'),price_loss=('price_loss','mean'))
    monthly.corr().to_csv(out/'drift_error_correlations.csv')
    print('Stage 2 complete. Selected using development period:',final,flush=True)

if __name__=='__main__':
    main()
