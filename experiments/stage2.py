"""Stage 2: calibrate, forecast, observe, and optionally replace tomorrow's model.

The default reproduces the original 1,095-day experiment. Exploratory windows
must write to a separate --output-dir; forecast artifacts are never modified.
"""
from pathlib import Path
import argparse
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


def day_types(df, output_dir=None):
    out=Path(output_dir) if output_dir is not None else ROOT/'results'
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
    profiles.to_csv(out/'cluster_profiles.csv')
    shares=pd.crosstab(days.year,labels,normalize='index')*100
    shares.index.name='year'
    shares.to_csv(out/'cluster_yearly_shares.csv')
    daily.to_csv(out/'day_types.csv',index_label='day')
    pd.DataFrame(scores).to_csv(out/'cluster_selection.csv',index=False)
    metadata={'normalisation':'divide each daily demand profile by its mean; shape only',
              'fit_period':'2010-2013 only; never refitted','pca_components':int(pca.n_components_),
              'explained_variance':float(pca.explained_variance_ratio_.sum()),
              'selected_k':best,'unusual_distance_cutoff_training_99pct':float(cutoff)}
    (out/'cluster_config.json').write_text(json.dumps(metadata,indent=2))
    joblib.dump({'pca':pca,'kmeans':km},out/'day_type_models.joblib')
    print('Day types:',metadata,flush=True)
    return daily


def input_statistics(df):
    """Monitor inputs available at the end of day D, without using targets or errors."""
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
    """Flag a rise in the rolling mean after seven confirmations; wait 540 days between alarms.
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


def strategy_name(policy, window_days):
    """Keep original three-year output names while labelling candidate windows."""
    if policy == 'frozen':
        return policy
    suffix = f'{window_days // 365}y' if window_days % 365 == 0 else f'{window_days}d'
    return f'{policy}_{suffix}'


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--window-days', type=int, default=WINDOW_DAYS)
    args = parser.parse_args(argv)
    if args.window_days <= 0:
        parser.error('--window-days must be positive')
    if args.window_days != WINDOW_DAYS and args.output_dir.resolve() == (ROOT/'results').resolve():
        parser.error('Candidate windows require a separate --output-dir to preserve baseline results')
    return args


def simulate_policy(*, x, target, days, valid, initial_model, model_factory,
                    input_points, input_threshold, error_reference, task, policy,
                    window_days=WINDOW_DAYS, budget=BUDGET):
    """Run one task/policy in time order with a budget shared across all years.

    1. Deploy any pre-scheduled replacement using completed days only.
    2. Forecast today's 48 intervals before observing today's outcomes.
    3. Observe today's inputs/errors and update the online detectors.
    4. A triggered replacement can first forecast tomorrow.
    """
    if window_days <= 0 or not 0 <= budget <= BUDGET:
        raise ValueError('Use a positive window and a replacement budget between zero and six')
    if policy not in ('frozen', 'periodic', 'input_drift', 'error_drift'):
        raise ValueError(f'Unknown policy: {policy}')
    simulated = np.flatnonzero(days >= pd.Timestamp('2015-01-01'))
    if not len(simulated) or len(simulated) % 48:
        raise ValueError('Simulation requires complete market days from 2015 onward')
    day_positions = simulated.reshape(-1, 48)
    dates = days[simulated][::48]
    schedule = {pd.Timestamp(f'{year}-01-01') for year in (2016, 2018, 2020, 2022, 2024, 2026)}
    error_means, error_scale, error_threshold = error_reference
    input_detector = Detector(input_threshold)
    error_detector = Detector(error_threshold)
    for day, value in input_points.loc['2014'].tail(MONITOR_DAYS).items():
        input_detector.observe(day, value)
    model = initial_model
    predictions = np.empty(len(simulated))
    alarms, retrains = [], []

    def replace(cutoff, decision_day):
        # Exclusive cutoff: a replacement deployed on T can learn only through T-1.
        fit = valid & (days < cutoff) & (days >= cutoff-pd.Timedelta(days=window_days))
        if not fit.any():
            raise ValueError('No valid rows in replacement window')
        replacement = model_factory()
        replacement.fit(x.loc[fit], np.asarray(target)[fit])
        retrains.append({'decision_day': str(decision_day.date()),
                         'effective_day': str(cutoff.date()), 'number': len(retrains)+1,
                         'train_end': str(days[fit].max().date()), 'rows': int(fit.sum())})
        return replacement

    for day_number, (forecast_day, positions) in enumerate(zip(dates, day_positions)):
        # Scheduled deployment: the date was chosen in advance, not from today's error.
        if policy == 'periodic' and forecast_day in schedule and len(retrains) < budget:
            model = replace(forecast_day, forecast_day-pd.Timedelta(days=1))

        # Forecast first. Features use completed historical market days and known weather.
        prediction = model.predict(x.iloc[positions])
        predictions[day_number*48:(day_number+1)*48] = prediction

        # Only now observe outcomes and update monitoring at the end of this day.
        observed = np.asarray(target)[positions]
        loss = (float(np.mean(100*np.abs(observed-prediction)/observed)) if task == 'demand'
                else float(100*np.mean(observed != prediction)))
        input_stat, input_alarm = input_detector.observe(forecast_day, input_points.loc[forecast_day])
        error_stat, error_alarm = error_detector.observe(
            forecast_day, (loss-error_means.loc[forecast_day.month])/error_scale)
        for detector, statistic, alarm, threshold in [
                ('input', input_stat, input_alarm, input_threshold),
                ('error', error_stat, error_alarm, error_threshold)]:
            if alarm:
                alarms.append({'detector': detector, 'day': str(forecast_day.date()),
                               'statistic': statistic, 'threshold': threshold, 'daily_loss': loss})

        triggered = ((policy == 'input_drift' and input_alarm) or
                     (policy == 'error_drift' and error_alarm))
        if triggered and len(retrains) < budget and day_number < len(dates)-1:
            model = replace(forecast_day+pd.Timedelta(days=1), forecast_day)
            error_detector.reset_errors()
    return {'predictions': predictions, 'index': x.index[simulated],
            'alarms': alarms, 'retrains': retrains}


def select_development_candidates(summary):
    """Select solely on 2015-2024; exact ties prefer the original 1,095-day window."""
    selected = {}
    for task, metric, ascending in [('demand', 'MAPE', True), ('price', 'BalAcc', False)]:
        candidates = summary[(summary.task == task) & (summary.split == 'development')].copy()
        if candidates.empty or not np.isfinite(candidates[metric]).all():
            raise ValueError(f'Missing finite development scores for {task}')
        candidates['original_window'] = candidates.window_days != WINDOW_DAYS
        best = candidates.sort_values([metric, 'original_window', 'window_days', 'strategy'],
                                      ascending=[ascending, True, True, True]).iloc[0]
        selected[task] = {'window_days': int(best.window_days), 'strategy': best.strategy,
                          'metric': metric, 'development_score': float(best[metric])}
    return selected


def main():
    args=parse_arguments()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    stage1=ROOT/'results/stage1_frozen'
    # 1. Prepare the shared causal features and fixed-reference diagnostics.
    df,truth,baseline=prepare(args.data)
    x=features(df)
    days=h.market_day(df.index)
    daily_types=day_types(df,out)
    point,input_monitor,input_threshold=input_statistics(df)
    config=json.loads((stage1/'config.json').read_text())
    valid=x.notna().all(axis=1).to_numpy()
    train=valid & (days<'2014-01-01')
    validation=valid & (days>='2014-01-01') & (days<'2015-01-01')
    sim=days>='2015-01-01'
    all_metrics=[]
    summary=[]
    alarms=[]
    retrains=[]
    # Evaluate every frozen Stage 1 model, including those not selected.
    for family,task in [('ridge','demand'),('boost_demand','demand'),('logistic','price'),('boost_price','price')]:
        target='nsw_demand_mw' if task=='demand' else 'nsw_price_up'
        prediction=pd.read_parquet(stage1/f'{family}_simulation.parquet')[target]
        all_metrics.extend(period_metrics(prediction.index,truth.loc[prediction.index,target],prediction,task,family,'frozen'))
    for name,pred in baseline.items():
        for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
            all_metrics.extend(period_metrics(df.index[sim],truth.loc[sim,col],pred.loc[sim,col],task,name,'baseline'))
    # 2. Calibrate on 2014, then simulate each selected model's replacement policies.
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
            'retraining_budget':BUDGET,'training_window_days':args.window_days},indent=2))
        initial=joblib.load(stage1/f'{family}.joblib')
        for policy in ('frozen','periodic','input_drift','error_drift'):
            strategy=strategy_name(policy,args.window_days)
            result=simulate_policy(
                x=x,target=truth[col].to_numpy(),days=days,valid=valid,
                initial_model=initial,model_factory=lambda: make_model(family,setting),
                input_points=point,input_threshold=input_threshold,
                error_reference=(emean,escale,ethreshold),task=task,policy=policy,
                window_days=args.window_days)
            predictions=result['predictions']
            retrains.extend({'task':task,'strategy':strategy,**row} for row in result['retrains'])
            alarms.extend({'task':task,'strategy':strategy,**row} for row in result['alarms'])
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
    # 3. Save all policies separately; choose using development scores only.
    pd.DataFrame(all_metrics).to_csv(out/'stage2_period_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'adaptation_summary.csv',index=False)
    pd.DataFrame(alarms).to_csv(out/'drift_signals.csv',index=False)
    pd.DataFrame(retrains).to_csv(out/'retraining_log.csv',index=False)
    # Select using 2015-2024; evaluate the holdout without changing the policy.
    final={}
    table=pd.DataFrame(summary)
    for task in ('demand','price'):
        subset=table[(table.task==task)&(table.split=='development')]
        best=subset.sort_values('MAPE' if task=='demand' else 'BalAcc',ascending=task=='demand').iloc[0]
        final[task]={'strategy':best.strategy,**config['selected'][task]}
    (out/'final_strategy.json').write_text(json.dumps({'selection_period':'2015-2024 only','holdout':'2025-October 2026, no policy revision',
        'window_days':args.window_days,'selected':final},indent=2))
    # Compare day types and input alarms with frozen-model daily errors.
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
