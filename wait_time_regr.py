import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

def feature_lag(dt, lag):
    col_name = dt.name
    dtl = list(dt)
    output = dict()
    i = 0
    if lag == 0:
        return dt

    while i < lag:
        prev = dtl[:-1]
        prev.insert(0, prev[0])
        output[col_name + '_lag_' + str(i + 1)] = prev
        dtl = prev
        i = i + 1

    while i > lag:
        futr = dtl[1:]
        futr.append(futr[-1])
        output[col_name + '_fut_' + str(-i + 1)] = futr
        i = i - 1
        dtl = futr
    return pd.DataFrame(output, index=dt.index)


def mape(y_true, y_pred):
    ts = np.array([y_true, y_pred])
    idx = np.isnan(ts).any(axis=0)
    y_true = ts[0][~idx] + 0.001
    y_pred = ts[1][~idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


tik = pd.read_table(
    'file_tik.csv', sep=',')
agt = pd.read_table('file_agt.csv', sep=',')

# tik['mapped_lang'] = tik['dim_language'].apply(lambda x: lang_map(x))
tik['start_utc_ts'] = tik['ts_created_at'].apply(lambda x: ts_15b(x))
tik['ts_created_at'] = tik['ts_created_at'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.0'))
tik['start_utc_hr'] = tik['ts_created_at'].apply(
    lambda x: datetime.strftime(x, '%Y-%m-%d %H:%00'))
tik['ts_assigned_at'] = tik['ts_created_at'] + \
    tik['wt_in_minutes'].apply(lambda x: timedelta(minutes=x))

merged = pd.merge(tik, agt, on=['start_utc_ts', 'language', 'sector', 'region'])
df = merged.copy()
df['unassigned'] = 0
for idx in df.index:
    c_ts = df.loc[idx, 'ts_created_at']
    df.loc[idx, 'unassigned'] = sum(
         (df[df['ts_created_at'] < c_ts]['ts_assigned_at'] > c_ts))

# agg the data at hourly level
agged = df[['start_utc_hr', 'wt_in_minutes']].groupby(
        'start_utc_hr').agg(['median', 'count'])
agged.columns = ['wait_time', 'tot_tik']
act_schedule = df[['start_utc_hr', 'act_agt', 'mess_agt', 'unassigned']
                      ].groupby('start_utc_hr').mean()
agged = agged.join(act_schedule)
op = agged.copy()
op = op.join(feature_lag(op['tot_tik'], 2))
op = op.join(feature_lag(op['act_agt'], -2))

dt = op.copy()
y = dt['wait_time']
del dt['wait_time']
X = dt
l = (len(y) / 3) * 2

reg = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.2)
reg.fit(X[:l], y[:l])
print(reg.score(X[:l], y[:l]))
print(mape(y[l:], reg.predict(X[l:])))
dt['wt_pred'] = reg.predict(X)


iter = 10000
nn = MLPRegressor(
    hidden_layer_sizes=(400, 50, 20, ), solver='lbfgs', max_iter=iter, alpha=0.1, batch_size=5)

nn.fit(X[:l], y[:l])
print(nn.score(X[:l], y[:l]))
print(mape(y[l:], nn.predict(X[l:])))
dt['wt_pred_nn'] = nn.predict(X)

print("naive pred")
# print(mape(y[l:], dt['wait_time_lag_1'][l:]))

dt['wt_actual'] = y
dt_plt = dt[dt.index > '2018-01-20']

agged_plt = agged[agged.index > '2018-01-20']
fig, ax = plt.subplots()

ax.plot(agged_plt.index, agged_plt['wait_time'], color='red')
ax.set_ylim([0, 1000])
ax2 = ax.twinx()
ax2.plot(agged_plt.index, agged_plt['act_agt'], color='blue')
ax3 = ax.twinx()
ax3.plot(agged_plt.index, agged_plt['tot_tik'], color='green')
for i, label in enumerate(ax.get_xticklabels()):
    if i % 24 > 0:
        label.set_visible(False)
fig.autofmt_xdate()
plt.show()

fig, ax = plt.subplots()
ax.plot_date(dt_plt.index, dt_plt['wt_actual'], marker='None', color='r',
             ls='-', label='true wait time')
ax.plot_date(dt_plt.index, dt_plt['wt_pred'], marker='None', color='b',
             ls='-', label='lm pred WT')
ax.plot_date(dt_plt.index, dt_plt['wt_pred_nn'], marker='None', color='k',
             ls='-', label='nn pred WT')
ax.legend()
fig.suptitle('comparing actual hourly median  WAIT TIME with predicted value')
plt.show()
