
import pandas as pd

def describe_all(df):
    
    def p1(x):
        return x.quantile(0.01)
    def p5(x):
        return x.quantile(0.05)
    def p25(x):
        return x.quantile(0.25)
    def p50(x):
        return x.quantile(0.5)
    def p75(x):
        return x.quantile(0.75)
    def p95(x):
        return x.quantile(0.95)
    def p99(x):
        return x.quantile(0.99)

    out = df.agg(["dtype","nunique"]).T
    out = pd.merge(
        out,
        (df.isnull().sum()/len(df)).to_frame(name="missing"),
        left_index=True,right_index=True
    )
    out = pd.merge(
        out,
        df.describe(exclude='number').T[['top','freq']],
        how='left',left_index=True,right_index=True
    )
    out = pd.merge(
        out,
        df.select_dtypes(include='number').agg(["min",p1,p5,p25,p50,p75,p95,p99,"max","mean","std"]).T,
        how='left',left_index=True,right_index=True
    )

    return out