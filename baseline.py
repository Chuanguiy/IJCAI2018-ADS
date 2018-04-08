import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import time

# global data
drop_list = []


def timestamp_datetime(value):
    tm_format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(tm_format, value)
    return dt


def extract_time(data):
    data['context_timestamp'] = data['context_timestamp'].apply(timestamp_datetime)
    data['context_timestamp'] = pd.to_datetime(data['context_timestamp'])

    data['day'] = data['context_timestamp'].dt.day
    data['hour'] = data['context_timestamp'].dt.hour
    data['hour'] = data['hour'].apply(map_hour)
    return data.drop('context_timestamp', axis=1)


def map_hour(x):
    if (x >= 7) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 20):
        return 2
    return 3


def process_on_hour(data):
    data['hour_map'] = data['hour'].apply(map_hour)
    return data.drop('hour', axis=1)


def category_trans(col):
    values = col.unique()
    tran = col.copy()
    for i in xrange(len(values)):
        tran[col == values[i]] = i
    tran.astype(int)
    return tran


def user_info_process(data):
    data['user_gender_id'] = data['user_gender_id'].map({-1: -1, 0: 0, 1: 1, 2: 0}).astype(int)
    data['user_gender_id'] += 1

    data.loc[data['user_age_level'] < 1002, 'user_age_level'] = 0
    data.loc[data['user_age_level'] > 1003, 'user_age_level'] = 2
    data.loc[data['user_age_level'] > 1000, 'user_age_level'] = 1

    data['user_occupation_id'] = data['user_occupation_id'].map({-1: 0, 2003: 0, 2002: 1, 2004: 1, 2005: 1}).astype(int)

    data.loc[data['user_star_level'] < 3001, 'user_star_level'] = 0
    data.loc[data['user_star_level'] > 3008, 'user_star_level'] = 2
    data.loc[data['user_star_level'] > 2999, 'user_star_level'] = 1

    return data


def get_category_list(data):
    #   data['category_0'] = data['item_category_list'].apply(lambda x: str(x).split(';')[0])
    data['category_1'] = data['item_category_list'].apply(lambda x: str(x).split(';')[1])
    data['category_2'] = data['item_category_list'].apply(
        lambda x: str(x).split(';')[2] if len(str(x).split(';')) > 2 else 'None')
    data = data.drop('item_category_list', axis=1)

    for col in ['category_1', 'category_2']:
        values = data[col].unique()
        data[col] = category_trans(data[col])

    data['category_2'] = data['category_2'] == 0
    data['category_1'].replace([3, 5, 8, 9], 3, inplace=True)
    data['category_1'].replace([7, 10, 11, 12], 7, inplace=True)

    data['category_1'] = category_trans(data['category_1'])
    return data


def get_property_list(data):
    data['property_0'] = data['item_property_list'].apply(lambda x: str(x).split(";")[0])
    data['property_1'] = data['item_property_list'].apply(lambda x: str(x).split(";")[1])
    data['property_2'] = data['item_property_list'].apply(
        lambda x: str(x).split(';')[2] if len(str(x).split(';')) > 2 else 'None')
    data = data.drop('item_property_list', axis=1)

    for col in ['property_'+str(x) for x in xrange(3)]:
        pass
    return data


def item_info_process(data):
    data = get_category_list(data)
    # data = get_property_list(data)

    return data


def get_brand_info(data):
    # get the features of item brand
    nums_item = data.groupby('item_brand_id', as_index=False)['item_id'].agg(
        {'item_nums_brand': lambda x: len(x.unique())})
    nums_shop = data.groupby('item_brand_id', as_index=False)['shop_id'].agg(
        {'shop_nums_brand': lambda x: len(x.unique())})
    nums_ins = data.groupby('item_brand_id', as_index=False)['instance_id'].agg({'instance_nums_brand': 'count'})
    nums_user = data.groupby('item_brand_id', as_index=False)['user_id'].agg(
        {'user_nums_brand': 'count', 'fans_nums_brand': lambda x: len(x.unique())})
    # nums_trade = data.groupby('item_brand_id', as_index=False)['is_trade'].agg(
    # {'trade_ratio_brand': lambda x: x.sum()})
    avg_collected = data.groupby('item_brand_id', as_index=False)['item_collected_level'].agg(
        {'avg_collected_brand': lambda x: x.sum()/x.shape[0]})

    brand_info = None
    for tmp in [nums_item, nums_shop, nums_ins, nums_user, avg_collected]:
        if brand_info is None:
            brand_info = tmp
        else:
            brand_info = pd.merge(left=brand_info, right=tmp, how='left', on='item_brand_id')

    # brand_info.sort_values('trade_ratio_brand', ascending=False, inplace=True)
    brand_info['item_per_shop'] = brand_info['item_nums_brand'] / brand_info['shop_nums_brand']
    # normalization
    for col in brand_info.columns:
        if col not in['item_brand_id', 'item_per_shop']:
            mean = brand_info[col].mean()
            std = brand_info[col].std()
            brand_info[col] = (brand_info[col] - mean) / std

    brand_info['brand_label'] = KMeans(n_clusters=7, random_state=10).fit_predict(brand_info.drop('item_brand_id', axis=1))
    brand_info = dummy(brand_info, ['brand_label'])

    return brand_info


def get_shop_info(data):
    base_cols = ['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
                 'shop_score_delivery', 'shop_score_description']
    shop_base = data[['shop_id']+base_cols].copy()
    shop_base.drop_duplicates(['shop_id'], inplace=True)

    # get the features of shop
    nums_item = data.groupby('shop_id', as_index=False)['item_id'].agg({'item_nums_shop': lambda x: len(x.unique())})
    nums_brand = data.groupby('shop_id', as_index=False)['item_brand_id'].agg(
        {'brand_nums_shop': lambda x: len(x.unique())})
    nums_ins = data.groupby('shop_id', as_index=False)['instance_id'].agg({'instance_nums_shop': 'count'})
    nums_user = data.groupby('shop_id', as_index=False)['user_id'].agg(
        {'user_nums_shop': 'count', 'fans_nums_shop': lambda x: len(x.unique())})
    shop_info = None
    for tmp in [nums_item, nums_brand, nums_ins, nums_user, shop_base]:
        if shop_info is None:
            shop_info = tmp
        else:
            shop_info = pd.merge(left=shop_info, right=tmp, how='left', on='shop_id')

    shop_info['item_brand'] = shop_info['item_nums_shop'] / shop_info['brand_nums_shop']
    #
    # print shop_info.head(10)

    for col in shop_info.columns:
        if col not in['shop_id', 'item_brand', 'shop_review_positive_rate', 'shop_score_service',
                      'shop_score_delivery', 'shop_score_description']:
            mean = shop_info[col].mean()
            std = shop_info[col].std()
            shop_info[col] = (shop_info[col] - mean) / std

    shop_info['shop_label'] = KMeans(n_clusters=3, random_state=10).fit_predict(shop_info.drop('shop_id', axis=1))
    shop_info = dummy(shop_info, ['shop_label'])

    shop_info.drop(base_cols, axis=1, inplace=True)
    print shop_info.info()
    return shop_info


def id_transform(train, test, ids):
    concated = pd.concat([train, test], axis=0)

    for id in ids:
        sumer = concated.groupby(id)[['is_trade']].sum()+1
        counter = concated.groupby(id)[['is_trade']].count()+10
        ratio = sumer / counter
        print type(ratio)
        ratio = ratio.reset_index()
        print ratio.describe()
        print ratio.shape
        print ratio.head(5)

        ratio.columns = [id, id+'_ratio']
        concated = pd.merge(left=concated, right=ratio, how='left', on=id)

    concated.drop(ids, axis=1, inplace=True)

    train = concated[:train.shape[0]]
    test = concated[train.shape[0]:]

    return train, test


def dummy(data, columns):
    for column in columns:
        if column not in data.columns:
            continue
        dummy_data = pd.get_dummies(data[column], drop_first=False)
        # rename columns: column name + 1,2,3
        c_num = len(dummy_data.columns)

        dummy_data.columns = [column + str(x + 1) for x in range(c_num)]
        data = pd.concat([data, dummy_data], axis=1)

        data = data.drop(column, axis=1)
    return data


def normalization(data, cols):
    for col in cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    return data


def data_process(data):
    data = extract_time(data)
    data = user_info_process(data)
    data = item_info_process(data)
    #   data = shop_info_process(data)
    data.drop(drop_list, 1, inplace=True)
    return data


def process_on_concat(train, test, process, *args, **keys):
    concated = pd.concat([train.drop('is_trade', 1), test], axis=0)
    concated = process(concated, *args, **keys)

    train_part = concated[:train.shape[0]]
    train = pd.concat([train_part, train['is_trade']], axis=1)

    test = concated[train.shape[0]:]
    return train, test


def log_loss(label, proba):
    loss = 0.0
    loss += np.sum(np.log(proba[label == 1]))
    loss += np.sum(np.log(1 - proba[label == 0]))
    return loss / len(proba)


def cross_valid(train, model, not_used):
    # get the used data
    used_col = [col for col in train.columns if col not in not_used]
    train = train[used_col]
    X, y = train.drop('is_trade', 1), train['is_trade']
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=False)
    skf.get_n_splits(X, y)
    epoch = 0
    loss = 0.0
    for train_index, valid_index in skf.split(X, y):
        Xtr, Xva = X.iloc[train_index], X.iloc[valid_index]
        ytr, yva = y.iloc[train_index], y.iloc[valid_index]
        model.fit(Xtr, ytr)
        pred = model.predict_proba(Xva)
        pred = pred[:, -1]
        # print pred.shape, yva.shape
        logloss = log_loss(yva, pred)
        loss += logloss
        print '[%d] log loss - %.4f' % (epoch, logloss)
        epoch += 1

    print '[cross valid] mean of log loss - %.4f' % (loss / 5)


def last_day_valid(data, model, not_used):
    # train valid split
    train = data[data['day'] < 24]
    valid = data[data['day'] == 24]
    # get the used data
    used_col = [col for col in train.columns if col not in not_used]
    train = train[used_col]
    valid = valid[used_col]
    # train and valid
    model.fit(train.drop('is_trade', 1), train['is_trade'])
    pred = model.predict_proba(valid.drop('is_trade', 1))
    pred = pred[:, -1]
    loss = log_loss(valid['is_trade'], pred)
    print '[last day valid] log loss - %.4f' % loss


def check_df_format(data):
    print type(data)
    print data.shape
    print data.info()


def submit(train, test, model, not_used):
    ins_id = test['instance_id']
    used_col = [col for col in train.columns if col not in not_used]
    train = train[used_col]
    used_col.remove('is_trade')
    feat = test[used_col]

    date = time.localtime(time.time())
    date = str(date.tm_mon) + '-' + str(date.tm_mday)
    model.fit(train.drop('is_trade', 1), train['is_trade'])

    pred = model.predict_proba(feat)

    test['predicted_score'] = pred[:, 1]
    test['instance_id'] = ins_id.values

    sub = test[['instance_id', 'predicted_score']]

    sub.to_csv('output/submit%s.txt' % date, index=False, sep=" ")


def get_test_trade_by_lr(train, test, not_used):
    used_col = [col for col in train.columns if col not in not_used]
    train = train[used_col]
    used_col.remove('is_trade')
    test = test[used_col]
    model = LogisticRegression()
    model.fit(train.drop('is_trade', 1), train['is_trade'])
    pred = model.predict(test)
    return pred


if __name__ == "__main__":
    no_used = ['instance_id', 'shop_id', 'user_id', 'item_id', 'item_city_id', 'item_brand_id', 'day',
                'item_category_list', 'item_property_list', 'predict_category_property',
                ]
    ids = ['shop_id', 'item_id']

    train_df = pd.read_table('input/round1_ijcai_18_train_20180301.txt', sep='\s+')
    test_df = pd.read_table('input/round1_ijcai_18_test_a_20180301.txt', sep='\s+')
    concated = pd.concat([train_df.drop('is_trade', 1), test_df])
    # brand cluster
    brand_df = get_brand_info(concated)
    shop_df = get_shop_info(concated)

    result = test_df['instance_id']

    # feature extraction
    train_df = data_process(train_df)
    test_df = data_process(test_df)
    ohe_list = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    norm_list = ['item_price_level', 'item_sales_level', 'item_collected_level', 'shop_star_level']

    train_df, test_df = process_on_concat(train_df, test_df, dummy, ohe_list)
    train_df, test_df = process_on_concat(train_df, test_df, normalization, norm_list)
    # merge data
    train_df = pd.merge(left=train_df, right=brand_df, how='left', on='item_brand_id')
    train_df = pd.merge(left=train_df, right=shop_df, how='left', on='shop_id')

    test_df = pd.merge(left=test_df, right=brand_df, how='left', on='item_brand_id')
    test_df = pd.merge(left=test_df, right=shop_df, how='left', on='shop_id')
    # get the ratio of each id

    test_df['is_trade'] = get_test_trade_by_lr(train_df, test_df, no_used)

    # train_df, test_df = id_transform(train_df, test_df, ids)
    test_df.drop('is_trade', axis=1, inplace=True)
    print train_df.columns

    model = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=1000)
    # evaluate
    cross_valid(train_df, model, no_used)

    # predict and submission
    last_day_valid(train_df, model, no_used)
    submit(train_df, test_df, model, no_used)
