# 예측할 데이터가 있는 경로, 모델 경로, 예측 결과를 저장할 경로를 받아서 예측 수행
import numpy as np
import xgboost as xgb

class predictor:
        
    def predict_lgb(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test, num_iteration=model.best_iteration)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        return preds_mean
        
    def predict_xgb(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        return preds_mean
        
    def predict_rf(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        return preds_mean
        
    def predict_mlpr(self, models:list, test):
        preds = []
        
        for model in models:
            
            pred = model.predict(test)
            preds.append(pred)
        
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array, axis=0)
        return preds_mean