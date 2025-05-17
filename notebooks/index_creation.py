

corr = df.select_dtypes("float").rename(columns={
    "bank orders": "volume bank orders",
    "The number of accounts opened by institutions of the banking system, per 1 resident, units.": "accounts per 1",
    "Number of bank institutions per 1 million inhabitants": "bank_isnt_per_1",
    "Total number of banking system institutions": "total_bank",
    "volume orders in electronic form with internet": "vol_internet",
    "payment requirements with internet": "pay_internet"
}).corr("spearman")

sns.heatmap(corr, annot=True, linewidths=.5,)


import numpy as np

for col in df.drop(labels=["region", "date"], axis=1).columns:
    df[f'{col}_normalized'] = (
        df.groupby(['date'])[col]
        .transform(
            lambda x: (lambda tx: (tx - tx.min()) / (tx.max() - tx.min() + 1e-8))( 
                np.where(x >= 1, np.log(x), 0)
            )
        )
    )


def geometric_mean_no_zeros(row):
    positive_values = row[row > 0]
    if len(positive_values) == 0:
        return 0
    return gmean(positive_values)

df['geometric_mean'] = df[df.columns[df.columns.str.contains("_normalized")]].apply(geometric_mean_no_zeros, axis=1)



df = df.sort_values(by=["date", "region"])
df.index = df.date

import numpy as np
import pandas as pd

# Load NPSDI data (0-1 scale)
npsdi = index_2023

# Calculate deciles using pandas.qcut
npsdi['decile'] = pd.qcut(npsdi['geometric_mean'], q=10, labels=False) + 1

# Handle ties via average ranking
npsdi['rank'] = npsdi['geometric_mean'].rank(method='average')
npsdi['percentile'] = (npsdi['rank'] - 0.5) / len(npsdi) * 100
npsdi['decile'] = np.floor(npsdi['percentile'] / 10) + 1


N = len(factors)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]



values =df_mosc.mean().values.tolist()
values += values[:1]

values_st =df_st.mean().values.tolist()
values_st += values_st[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

ax.plot(angles, values, color='blue', linewidth=2, label='Moscow')
ax.plot(angles, values_st, color='green', linewidth=2, label='St. Petersburg')

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.legend(loc='best')
plt.xticks(angles[:-1], factors)
plt.title('Top Regions', y=1.08)
plt.ylim(0, 1 * 1.1)



import matplotlib.pyplot as plt

N = len(factors)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
values =df_mosc.mean().values.tolist()
values += values[:1]

values_st =df_st.mean().values.tolist()
values_st += values_st[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

ax.plot(angles, values, color='blue', linewidth=2, label='Chechen')
ax.plot(angles, values_st, color='green', linewidth=2, label='Republic of Dagestan')

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.legend(loc='best')
plt.xticks(angles[:-1], factors)
plt.title('Losers Regions', y=1.08)
plt.ylim(0, 1 * 1.1)





