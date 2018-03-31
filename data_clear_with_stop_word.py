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


def get_data():
    train = pd.read_csv("train_first.csv")
    test = pd.read_csv("predict_first.csv")
    train.Discuss.fillna('_na_', inplace=True)
    test.Discuss.fillna('_na_', inplace=True)
    data = pd.concat([train, test])
    return data


def splitWord(query, stopwords):
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result


def process(data):
    stopwords = {}
    for line in codecs.open('stop.txt', 'r', 'utf-8'):
        stopwords[line.rstrip()] = 1
    data['doc'] = data['Discuss'].map(lambda x: splitWord(x, stopwords))
    return data


# data, train, test = get_data()

# train = process(train)
# test = process(test)


def vec_tf_fit():
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    data = get_data()
    data = process(data)
    test_data = data.loc[data.Score.isnull(), :]
    target = data.loc[data.Score.notnull(), "Score"].values
    assert isinstance(data, pd.DataFrame)
    pipe_line = Pipeline([("CountVector", CountVectorizer(ngram_range=(1, 2), min_df=5)),
                          ("TfIdf", TfidfTransformer())])
    trans = pipe_line.fit_transform(data['doc'].values)
    train = trans[data['Score'].notnull().values, :]
    test = trans[data['Score'].isnull().values, :]
    x_train, x_test, y_train, y_test = train_test_split(train,
                                                        target,
                                                        test_size=0.3, random_state=0)
    xgb_clf = XGBClassifier(max_depth=5, silent=True,
                            subsample=0.8, colsample_bytree=0.7, n_jobs=4,
                            scale_pos_weight=2)
    xgb_clf.fit(train, target)
    rf_clf = RandomForestClassifier(n_estimators=150,
                                    max_depth=20,
                                    max_features=0.2,
                                    min_samples_split=5,
                                    min_samples_leaf=4,
                                    class_weight="balanced",
                                    n_jobs=4,
                                    random_state=0,
                                    oob_score=False)
    rf_clf.fit(train, target)

    score = cross_val_score(rf_clf, train, target, cv=3, n_jobs=4)
    print(score)
    print(classification_report(target, rf_clf.predict(train)))
    test_data['Score'] = np.asarray(xgb_clf.predict(test))
    test_data.to_csv("xgboost_stop_word.csv")


if __name__ == '__main__':
    vec_tf_fit()
