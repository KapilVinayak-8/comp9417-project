"""Small controlled feature experiments on 2014 only, plus descriptive data tables."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forecast.model import features, WEATHER
from experiments.baselines import prepare
from experiments.stage1 import make_model
from harness import dayahead as h


def main():
    df,truth,_=prepare()
    x=features(df)
    days=h.market_day(df.index)
    valid=x.notna().all(axis=1).to_numpy()
    train=valid & (days<'2014-01-01')
    val=valid & (days>='2014-01-01') & (days<'2015-01-01')
    selected=json.loads((ROOT/'results/stage1_frozen/config.json').read_text())['selected']
    results=[]
    importance=[]
    for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
        for ablation in ('all_features','no_other_states','no_weather','no_market_lags'):
            cols=list(x.columns)
            if ablation=='no_other_states': cols=[c for c in cols if not c.startswith(('qld_','vic_','mel_','bne_'))]
            if ablation=='no_weather': cols=[c for c in cols if not c.startswith(('syd_','mel_','bne_'))]
            if ablation=='no_market_lags': cols=[c for c in cols if not c.startswith(('nsw_','qld_','vic_'))]
            model=make_model(selected[task]['family'],selected[task]['setting'])
            model.fit(x.loc[train,cols],truth.loc[train,col])
            pred=model.predict(x.loc[val,cols])
            y=truth.loc[val,col]
            metrics={'MAPE':h.mape(y,pred),'MAE':h.mae(y,pred)} if task=='demand' else {'Acc':h.accuracy(y,pred),'BalAcc':h.balanced_accuracy(y,pred)}
            results.append({'task':task,'variant':ablation,'features':len(cols),**metrics})
            if ablation=='all_features':
                for c,value in zip(cols,model.feature_importances_):
                    importance.append({'task':task,'feature':c,'split_importance':int(value)})
            print(task,ablation,metrics,flush=True)
        if task=='price':
            model=make_model(selected[task]['family'],selected[task]['setting'])
            model.set_params(class_weight='balanced')
            model.fit(x.loc[train],truth.loc[train,col])
            p=model.predict(x.loc[val])
            results.append({'task':task,'variant':'class_weight_balanced','features':x.shape[1],
                            'Acc':h.accuracy(truth.loc[val,col],p),'BalAcc':h.balanced_accuracy(truth.loc[val,col],p)})
    # These follow-up ablations explain the fixed Stage 1 choice; they do not revise it.
    pd.DataFrame(results).to_csv(ROOT/'results/feature_ablations.csv',index=False)
    pd.DataFrame(importance).to_csv(ROOT/'results/feature_importance.csv',index=False)
    df.describe(percentiles=[.01,.5,.99]).to_csv(ROOT/'results/data_summary.csv')
    daily=pd.DataFrame({'mean_price':df.nsw_price.groupby(days).mean(),
                        'max_price':df.nsw_price.groupby(days).max(),
                        'demand_mean':df.nsw_demand.groupby(days).mean(),
                        'positive_pct':truth.nsw_price_up.groupby(days).mean()*100})
    daily['previous_mean_price']=daily.mean_price.shift(1)
    daily.iloc[1:].to_csv(ROOT/'results/daily_class_balance.csv',index_label='day')
    yearly=df.groupby(days.year).agg(demand_mean=('nsw_demand','mean'),price_mean=('nsw_price','mean'),
                                   price_max=('nsw_price','max'),negative_price_pct=('nsw_price',lambda a:100*(a<0).mean()))
    yearly['positive_pct']=truth.nsw_price_up.groupby(days.year).mean()*100
    yearly.to_csv(ROOT/'results/yearly_data_summary.csv',index_label='year')

if __name__=='__main__': main()
