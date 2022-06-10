# 학습 데이터 경로, 모델 저장 경로, 하이퍼 파라미터들 받아서 모델 학습 및 저장
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.neural_network import MLPRegressor as mlpr
import lightgbm as lgb
from xgboost import XGBRegressor as xgbr
import xgboost as xgb

class trainer:
    def __init__(self):
        self.models_lgb = []
        self.models_xgbr = []
        self.models_xgb = []
        self.models_rf = []
        self.models_mlpr = []
        
        self.rmses_lgb = []
        self.rmses_xgbr = []
        self.rmses_xgb = []
        self.rmses_rf = []
        self.rmses_mlpr = []
        
        self.oof_lgb = []
        self.oof_xgbr = []
        self.oof_xgb = []
        self.oof_rf = []
        self.oof_mlpr = []
        
    def train_lgb(self, train_X, train_Y, params:dict={}, folds=5):
        models = []
        oof_lgb = np.zeros(len(train_X))
        
        kf = KFold(n_splits=folds)
        
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
            
            model_lgb = lgb.train(params=params,
                                train_set=lgb_train,
                                valid_sets=lgb_eval,
                                num_boost_round=100,
                                early_stopping_rounds=20,
                                verbose_eval=True,
                                )
            
            y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
            
            models.append(model_lgb)
            oof_lgb[val_index] = y_pred
            
        self.models_lgb.append(models)
        self.oof_lgb.append(oof_lgb)
        rmse_lgb = np.sqrt(mean_squared_error(train_Y, oof_lgb))
        self.rmses_lgb.append(rmse_lgb)
        
    def train_xgbr(self, train_X, train_Y, params:dict={}, folds=5):
        models = []
        oof_xgbr = np.zeros(len(train_X))
        
        kf = KFold(n_splits=folds)
        
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            
            model_xgbr = xgbr(n_estimators=300, **params)
            model_xgbr.fit(X=X_train, y=y_train)
            
            y_pred = model_xgbr.predict(X_valid)
            
            models.append(model_xgbr)
            oof_xgbr[val_index] = y_pred
            
        self.models_xgbr.append(models)
        self.oof_xgbr.append(oof_xgbr)
        rmse_xgbr = np.sqrt(mean_squared_error(train_Y, oof_xgbr))
        self.rmses_xgbr.append(rmse_xgbr)
        
    def train_xgb(self, train_X, train_Y, params:dict={}, folds=5):
        models = []
        oof_xgb = np.zeros(len(train_X))
        
        kf = KFold(n_splits=folds)
        
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            xgb_train = xgb.DMatrix(X_train, label=y_train)
            xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
            evals = [(xgb_train,'train'), (xgb_eval,'eval')]
            model_xgb = xgb.train(params,
                                xgb_train,
                                evals=evals,
                                num_boost_round=300,
                                early_stopping_rounds=20,
                                verbose_eval=20,
                                )
            y_pred = model_xgb.predict(xgb_eval)
            
            models.append(model_xgb)
            self.models_xgb.append(models)
            oof_xgb[val_index] = y_pred
            
        self.models_xgb.append(models)
        self.oof_xgb.append(oof_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(train_Y,oof_xgb))
        self.rmses_xgb.append(rmse_xgb)
        
    def train_rf(self, train_X, train_Y, folds=5):
        models = []
        oof_rf = np.zeros(len(train_X))

        kf = KFold(n_splits=folds)
        
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            
            model_rf = rf(n_estimators=50,
                        random_state=1234,
                        )
            model_rf.fit(X_train, y_train)
            y_pred = model_rf.predict(X_valid)
            
            models.append(model_rf)
            oof_rf[val_index] = y_pred
            
        self.models_rf.append(models)
        self.oof_rf.append(oof_rf)
        rmse_rf = np.sqrt(mean_squared_error(train_Y, oof_rf))
        self.rmses_rf.append(rmse_rf)
    
    def train_mlpr(self, train_X, train_Y, folds=5, params:dict={}):
        models = []
        oof_mlpr = np.zeros(len(train_X))
        
        kf = KFold(n_splits=folds)
        
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            
            model_mlpr = mlpr(**params)
            model_mlpr.fit(X_train, y_train)
            y_pred = model_mlpr.predict(X_valid)
            
            models.append(model_mlpr)
            self.models_mlpr.append(models)
            oof_mlpr[val_index] = y_pred
            
        self.models_mlpr.append(models)
        self.oof_mlpr.append(oof_mlpr)
        rmse_mlpr = np.sqrt(mean_squared_error(train_Y,oof_mlpr))
        self.rmses_mlpr.append(rmse_mlpr)