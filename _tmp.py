# -*- coding: utf-8 -*-
"""
@Time ： 2020/5/31 下午5:45
@Auth ： shenhao
@Email： shenhao@xiaomi.com
"""

import pandas as pd
import numpy as np

df = pd.DataFrame([['a1', 1, 5], ['a1', 4, 5], ['a1', 1, 6]], columns=['uid', 'score', 'salary'])

print(df[['score', 'salary']].groupby(by=df['uid']).agg(lambda x: np.mean(x.mode()[0])).reset_index())

df.to_csv('submission/result.csv', sep=',', index=False)