import pandas as pd
import numpy as np
from model.train import trainer
from model.predict import predictor
import yaml
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# 데이터 로딩
train_df = pd.read_csv(dir_path +'/mydata/train_df.csv')
test1_df = pd.read_csv(dir_path +'/mydata/test1_df.csv')
test2_df = pd.read_csv(dir_path +'/mydata/test2_df.csv')

submission = pd.read_csv(dir_path +'/submit/submit.csv')

train_X = train_df.drop(['date','heat_supply'], axis=1)
train_Y = train_df['heat_supply']

test1_X = test1_df.drop(['date', 'heat_supply'],axis=1)
test2_X = test2_df.drop(['date', 'heat_supply'],axis=1)


with open(dir_path + '/model/setting/params.yml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# 모델 학습
tr = trainer()
pred = predictor()

tr.train_lgb(train_X, train_Y, params['lgb'])
tr.train_xgb(train_X, train_Y, params['xgb'])
# tr.train_xgbr(train_X, train_Y, params['xgbr'])
# tr.train_mlpr(train_X, train_Y, params=params['mlpr'])

# 예측
pred.predict_lgb(tr.models_lgb[0], test1_X)
pred.predict_lgb(tr.models_lgb[0], test2_X)

pred.predict_xgb(tr.models_xgb[0], test1_X)
pred.predict_xgb(tr.models_xgb[0], test2_X)
# pred.predict_xgbr(tr.models_xgbr[0],test_X)
# pred.predict_mlpr(tr.models_mlpr[0], test_X)

# 결과값 처리
preds_ans1_lgb = pred.preds_lgb[0].sum()
preds_ans2_lgb = pred.preds_lgb[1].sum()
preds_ans1_xgb = pred.preds_xgb[0].sum()
preds_ans2_xgb = pred.preds_xgb[1].sum()

# print('rmse of lgb:',tr.rmses_lgb)
# print('rmse of xgb:',tr.rmses_xgb) #[0.11601585076934848]
# print('rmse of xgbr:',tr.rmses_xgbr) #[0.11567734530963232]
# print('rmse of mlpr:',tr.rmses_mlpr) #[0.39306480765719265]

# weight_lgb = (1 / tr.rmses_lgb[0])
# weight_xgb = (1 / tr.rmses_xgb[0])
# weight_mlpr = (1 / tr.rmses_mlpr[0])

preds_final_ans1 = preds_ans1_lgb * 0.5 + preds_ans1_xgb * 0.5
preds_final_ans2 = preds_ans2_lgb * 0.5 + preds_ans2_xgb * 0.5

submission['heat_supply_sum'] = np.array([preds_final_ans1,preds_final_ans2])
submission.to_csv(dir_path +'/submit/220156.csv', index=False)