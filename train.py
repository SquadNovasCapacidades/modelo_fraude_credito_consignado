import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


def balance_class(Train):
    ### Balanceando a base de teste
    fraudes = Train.FLAG.sum()
    flag_1 = Train[Train.FLAG == 1]
    flag_0 = (Train[Train.FLAG == 0]).sample((fraudes*5)- fraudes)
    final_data = (pd.concat([flag_1, flag_0])).sample(frac=1)
    final_data = final_data.drop_duplicates()
    return final_data


def out_of_time_split(data, month):
    Teste = data[data.DATA_SOLICITACAO.str.contains(month)]
    Train = data[~data.DATA_SOLICITACAO.str.contains(month)]
    Train = Train.drop('DATA_SOLICITACAO', axis=1)
    Teste = Teste.drop('DATA_SOLICITACAO', axis=1)   
    Train = balance_class(Train)

    X_train = Train.drop(['FLAG'], axis=1).values
    y_train = Train.FLAG.values
    X_test = Teste.drop(['FLAG'], axis=1).values
    y_test = Teste.FLAG.values
    return [X_train, X_test, y_train, y_test]


def out_of_sample_split(data):
    X = data.drop('FLAG', axis=1).values
    y = data.FLAG.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 123, 
                                                        stratify = y)
    return [X_train, X_test, y_train, y_test]


def train_model(X_train, y_train):
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 60,
            'learning_rate': 0.05,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'max_depth':20
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=lgb_train)
    return gbm


def main_train(_file):
    data = pd.read_csv(_file, encoding='latin')
    X_train, X_test, y_train, y_test = out_of_time_split(data, 'MAR2017')
    model = train_model(X_train, y_train)
    joblib.dump(model, filename='modelo_fraude_credito_consignado.pkl')
    return True

if __name__ == '__main__':
    main_train(_file='output/prepared_data.csv')