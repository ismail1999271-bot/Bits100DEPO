"""Deterministic smoke demo: normalized OHLCV -> labels -> ranking -> metrics."""
from datetime import datetime, timedelta, timezone
from .pipeline import normalize_ohlcv, add_forward_labels, LabelConfig
from .daily_ranker import rank_latest

def rows():
    start = datetime(2026,1,1, tzinfo=timezone.utc)
    out=[]
    series={"AAA":[100,102,105,108,115],"BBB":[100,100,101,101,102],"CCC":[100,99,98,99,98]}
    for symbol, closes in series.items():
        for i,c in enumerate(closes):
            out.append({"symbol":symbol,"timestamp":start+timedelta(days=i),"open":c,"high":c,"low":c*.99,"close":c,"volume":1000+i*100})
    return out

def run():
    data=normalize_ohlcv(rows())
    labeled=add_forward_labels(data, LabelConfig(limit_pct=.05, horizon_bars=1))
    ranked=rank_latest(labeled)
    return {"rows":len(labeled),"symbols":labeled.symbol.nunique(),"ranked":ranked[["symbol","score"]].to_dict("records"),"forward_hits":int(labeled.hit_limit_forward.sum())}

if __name__ == "__main__": print(run())
