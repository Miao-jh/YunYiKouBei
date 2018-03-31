import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import codecs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

jieba.suggest_freq("不值得", tune=True)
# 解决不能正确分割出值得,不值得.....


def get_data():
    train = pd.read_csv("train_first.csv")
    test = pd.read_csv("predict_first.csv")
    train.Discuss.fillna('_na_', inplace=True)
    test.Discuss.fillna('_na_', inplace=True)
    data = pd.concat([train, test])
    return data


def split_word(query, stopwords):
    "jieba切割出来的词不在stopwords中"
    word_list = jieba.cut(query)
    num = 0
    result = ''
    for word in word_list:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result


def process(data: pd.DataFrame) -> pd.DataFrame:
    stopwords = {}
    for line in codecs.open('stop.txt', 'r', 'utf-8'):
        stopwords[line.rstrip()] = 1
    data['doc'] = data['Discuss'].map(lambda x: split_word(x, stopwords))
    return data


def class_weight(y) -> np.ndarray:
    "参考logistics regression 当中关于class_weight 的 balanced选项 打算用于xgboost的"
    le = LabelEncoder()
    y_ind = le.fit_transform(y)
    classes = np.unique(y)
    recip_freq = len(y) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
    weight = recip_freq[le.transform(classes)]
    return weight


def vec_tf_fit():
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    data = get_data()
    data = process(data)
    test_data = data.loc[data.Score.isnull(), :]  # 原始测试数据集
    target = data.loc[data.Score.notnull(), "Score"].values - 1  # 训练集label
    target = target.astype(int)
    weight = class_weight(target)
    sample_weight = weight[(target - 1).astype(int)]
    assert isinstance(data, pd.DataFrame)
    pipe_line = Pipeline([("CountVector", CountVectorizer(ngram_range=(1, 1), min_df=5)),
                          ("TfIdf", TfidfTransformer(norm='l2'))])
    trans = pipe_line.fit_transform(data['doc'].values)
    train = trans[data['Score'].notnull().values, :]
    test = trans[data['Score'].isnull().values, :]
    x_train, x_test, y_train, y_test = train_test_split(train,
                                                        target,
                                                        test_size=0.3, random_state=0)
    lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(x_test, y_test, free_raw_data=False)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclassova',
        'num_class': 5,
        'metric': 'multi_error',
        'num_leaves': 200,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': 0,
        'is_unbalance': True,
        'nthread': 4,
        'max_depth': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'min_data_in_leaf': 20,
    }
    gbm = lgb.train(lgb_params, lgb_train, valid_sets=lgb_eval,
                    num_boost_round=400, early_stopping_rounds=50)
    lgb_predict = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    lgb_score = lgb_predict.argmax(axis=1) + 1
    plt.hist(lgb_score)
    test_data['Score'] = lgb_score
    # xgb_clf = XGBClassifier(max_depth=15, silent=True,
    #                         subsample=0.2, colsample_bytree=0.2, n_jobs=4,
    #                         objective="multi:softmax", num_class=5, reg_alpha=0.5,
    #                         n_estimators=150, base_score=0.8, reg_lambda=0.5)
    # xgb_clf.fit(x_train, y_train, sample_weight=sample_weight)
    # rf_clf = RandomForestClassifier(n_estimators=300,
    #                                 max_depth=15,
    #                                 class_weight="balanced",
    #                                 n_jobs=4,
    #                                 random_state=0,
    #                                 oob_score=True)
    # rf_clf.fit(x_train, y_train)

    # score = cross_val_score(rf_clf, train, target, cv=3)
    # print(score)
    print(classification_report(y_test + 1, lgb_score, digits=5))
    # plt.hist(np.asarray(rf_clf.predict(test)))
    # test_data['Score'] = np.asarray(rf_clf.predict(test))
    test_data.to_csv("lgb_stop_word.csv")


if __name__ == '__main__':
    vec_tf_fit()
    """
                 precision    recall  f1-score   support

          1    0.08757   0.17127   0.11589       181
          2    0.06096   0.15667   0.08777       300
          3    0.23429   0.38348   0.29087      2858
          4    0.43832   0.40823   0.42274      8826
          5    0.73825   0.66134   0.69768     17835
avg / total    0.59130   0.55240   0.56843     30000

    """
