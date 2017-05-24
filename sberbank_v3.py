import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

df_train = pd.read_csv("train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

df_train.head()

# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
ylog_train_all = np.log1p(df_train['price_doc'].values)
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
df_all['max_floor'] = df_all['max_floor'].replace(to_replace=0, value=np.nan)
df_all['rel_life_sq'] = df_all['life_sq'] / df_all['full_sq'].astype(float)
df_all['rel_life_sq'] = df_all['rel_life_sq'].replace(to_replace=np.inf, value=np.nan) # Corrects for property with zero full_sq.
df_all['avg_room_sq'] = df_all['life_sq'] / df_all['num_room'].astype(float) # Does not account for living room, but reasonable enough.
df_all['avg_room_sq'] = df_all['avg_room_sq'].replace(to_replace=np.inf, value=np.nan) # Corrects for studios (zero rooms listed).



# Replace garbage values in build_year with NaNs, then find average build year
# in each sub_area.
df_all['build_year'] = df_all['build_year'].replace(to_replace=[0,1,2,3,20,71,215,4965], value=np.nan)
mean_by_districts = pd.DataFrame(columns=['district', 'avg_build_year'])
sub_areas_unique = df_all['sub_area'].unique()
for sa in sub_areas_unique:
    temp = df_all.loc[df_all['sub_area'] == sa]
    mean_build_year = temp['build_year'].mean()
    new_df = pd.DataFrame([[sa, mean_build_year]], columns=['district', 'avg_build_year'])
    mean_by_districts = mean_by_districts.append(new_df, ignore_index=True)

mbd_dis_list = mean_by_districts['district'].tolist()
mbd_dis_full = df_all['sub_area'].tolist()
mbd_aby_np = np.array(mean_by_districts['avg_build_year'])
mbd_aby_full = np.zeros(len(df_all.index))

# (Could find a better way to do this.)
for i in range(len(df_all.index)):
    district = mbd_dis_full[i]
    mbd_aby_full[i] = mbd_aby_np[mbd_dis_list.index(district)]

df_all['avg_build_year'] = mbd_aby_full
df_all['rel_build_year'] = df_all['build_year'] - df_all['avg_build_year']


# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)

# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

# Create a validation set, with last 20% of data
num_val = int(num_train * 0.2)

X_train_all = X_all[:num_train]
X_train = X_all[:num_train-num_val]
X_val = X_all[num_train-num_val:num_train]
ylog_train = ylog_train_all[:-num_val]
ylog_val = ylog_train_all[-num_val:]

X_test = X_all[num_train:]

df_columns = df_values.columns

print('X_train_all shape is', X_train_all.shape)
print('X_train shape is', X_train.shape)
print('y_train shape is', ylog_train.shape)
print('X_val shape is', X_val.shape)
print('y_val shape is', ylog_val.shape)
print('X_test shape is', X_test.shape)

dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)
dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'min_child_weight':1,
    'silent': 1,
    'seed':0
}

# Uncomment to tune XGB `num_boost_rounds`
partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                       early_stopping_rounds=20, verbose_eval=20)

num_boost_round = partial_model.best_iteration

num_boost_round = partial_model.best_iteration
model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)


ylog_pred = model.predict(dtest)
y_pred = np.exp(ylog_pred) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.to_csv('sub.csv', index=False)
