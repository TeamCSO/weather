import pandas as pd
import numpy as np
from model.train import trainer
from model.predict import predictor
import yaml
import joblib
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# ! 데이터 로딩
train_df = pd.read_csv(dir_path + "/data/df/train_df.csv")
test1_df = pd.read_csv(dir_path + "/data/df/test1_df.csv")
test2_df = pd.read_csv(dir_path + "/data/df/test2_df.csv")

submission = pd.read_csv(dir_path + "/submit/submit.csv")

train_X = train_df.drop(["Dates", "Date", "heat_supply"], axis=1)
train_Y = train_df["heat_supply"]

with open(dir_path + "/model/setting/params.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# ! 모델 학습
tr = trainer()
pred = predictor()

tr.train_lgb(train_X, train_Y, params["lgb"])
tr.train_xgb(train_X, train_Y, params["xgb"])
tr.train_ridge(train_X, train_Y, params["ridge"])

joblib.dump(tr.models_lgb, dir_path + "/data/model/model_lgb.pkl")
joblib.dump(tr.models_xgb, dir_path + "/data/model/model_xgb.pkl")
joblib.dump(tr.models_ridge, dir_path + "/data/model/model_ridge.pkl")

# ! 예측
# pred1 = pred.predict_lgb(tr.models_lgb, train_X)
# pred2 = pred.predict_xgb(tr.models_xgb, train_X)
# preds_ans1_lgb = pred.predict_lgb(tr.models_lgb, test1_X).sum()
# preds_ans2_lgb = pred.predict_lgb(tr.models_lgb, test2_X).sum()

# preds_ans1_xgb = pred.predict_xgb(tr.models_xgb, test1_X).sum()
# preds_ans2_xgb = pred.predict_xgb(tr.models_xgb, test2_X).sum()

print("rmse of lgb:", tr.rmse_lgb)
print("rmse of xgb:", tr.rmse_xgb)
print("rmse of ridge:", tr.rmse_ridge)

# ! 앙상블
# preds_final_ans1 = preds_ans1_lgb * 0.5 + preds_ans1_xgb * 0.5
# preds_final_ans2 = preds_ans2_lgb * 0.5 + preds_ans2_xgb * 0.5

# ! 결과값 저장
# submission['heat_supply_sum'] = np.array([preds_final_ans1,preds_final_ans2])
# submission.to_csv(dir_path +'/submit/220156.csv', index=False)
