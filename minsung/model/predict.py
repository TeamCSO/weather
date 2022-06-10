# 예측할 데이터가 있는 경로, 모델 경로, 예측 결과를 저장할 경로를 받아서 예측 수행
import numpy as np
import xgboost as xgb

class predictor:
    def __init__(self):
        self.preds_lgb = []
        self.preds_xgbr = []
        self.preds_xgb = []
        self.preds_rf = []
        self.preds_mlpr = []
        
    def predict_lgb(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test, num_iteration=model.best_iteration)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        self.preds_lgb.append(np.exp(preds_mean))
        
    def predict_xgbr(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        self.preds_xgbr.append(np.exp(preds_mean))
        
    def predict_rf(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        self.preds_rf.append(np.exp(preds_mean))
        
    def predict_xgb(self, models:list, test):
        preds = []
        xgb_test = xgb.DMatrix(data=test)
        for model in models:
            
            pred = model.predict(xgb_test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        self.preds_xgb.append(np.exp(preds_mean))
        
    def predict_mlpr(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        self.preds_mlpr.append(np.exp(preds_mean))