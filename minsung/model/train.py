# 학습 데이터 경로, 모델 저장 경로, 하이퍼 파라미터들 받아서 모델 학습 및 저장
from tabnanny import verbose
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.neural_network import MLPRegressor as mlpr
from sklearn.linear_model import Ridge
import lightgbm as lgb
from xgboost import XGBRegressor as xgbr
import xgboost as xgb

class trainer:
    
    # ! LightGBM
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
            
        self.models_lgb = models
        self.oof_lgb = oof_lgb
        self.rmse_lgb = np.sqrt(mean_squared_error(train_Y, oof_lgb))
        
    # ! XGBoost
    def train_xgb(self, train_X, train_Y, params:dict={}, folds=5):
        models = []
        oof_xgb = np.zeros(len(train_X))
        
        kf = KFold(n_splits=folds)
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            evals = [(X_valid, y_valid)]
            model_xgb = xgbr(**params, eval_metric='rmse')
            model_xgb.fit(X=X_train, y=y_train, eval_set=evals , verbose=20)
            
            y_pred = model_xgb.predict(X_valid)
            
            models.append(model_xgb)
            oof_xgb[val_index] = y_pred
            
        self.models_xgb = []
        self.models_xgb = models
        self.oof_xgb = oof_xgb
        self.rmse_xgb = np.sqrt(mean_squared_error(train_Y, oof_xgb))
        
    # ! Random Forest
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
            
        self.models_rf = models
        self.oof_rf = oof_rf
        self.rmse_rf = np.sqrt(mean_squared_error(train_Y, oof_rf))
        
    # ! Multi-layer Perceptron Regressor
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
            oof_mlpr[val_index] = y_pred
        
        self.models_mlpr = models
        self.oof_mlpr = oof_mlpr
        self.rmse_mlpr = np.sqrt(mean_squared_error(train_Y,oof_mlpr))
        
    # ! Ridge Regressor
    def train_ridge(self, train_X, train_Y, params:dict={}, folds=5):
        models = []
        oof_ridge = np.zeros(len(train_X))
        
        kf = KFold(n_splits=folds)
        
        for train_index, val_index in kf.split(train_X):
            X_train = train_X.iloc[train_index]
            X_valid = train_X.iloc[val_index]
            y_train = train_Y.iloc[train_index]
            y_valid = train_Y.iloc[val_index]
            
            model_ridge = Ridge(**params)
            model_ridge.fit(X_train, y_train)
            
            y_pred = model_ridge.predict(X_valid)
            
            models.append(model_ridge)
            oof_ridge[val_index] = y_pred
            
        self.models_ridge = models
        self.oof_ridge = oof_ridge
        self.rmse_ridge = np.sqrt(mean_squared_error(train_Y, oof_ridge))
        