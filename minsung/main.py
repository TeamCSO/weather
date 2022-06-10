import pandas as pd
import numpy as np
from model.train import trainer
from model.predict import predictor
import yaml
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

train_df = pd.read_csv(dir_path +'/mydata/train_df.csv')
# test_df = pd.read_csv(dir_path +'/mydata/test_df_test.csv')
submission = pd.read_csv(dir_path +'/mydata/submit.csv')

train_X = train_df.drop(['date','heat_supply'], axis=1)
train_Y = train_df['heat_supply']
# test_X = test_df.drop(['SalePrice','Id'], axis=1)

with open(dir_path + 'model/setting/params.yml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


tr = trainer()
pred = predictor()

tr.train_lgb(train_X, train_Y, params['lgb'])
# tr.train_xgb(train_X, train_Y, params['xgb'])
tr.train_xgbr(train_X, train_Y, params['xgbr'])
# tr.train_mlpr(train_X, train_Y, params=params['mlpr'])

# pred.predict_lgb(tr.models_lgb[0], test_X)
# pred.predict_xgb(tr.models_xgb[0], test_X)
# pred.predict_xgbr(tr.models_xgbr[0],test_X)
# pred.predict_mlpr(tr.models_mlpr[0], test_X)

# print('rmse of lgb:',tr.rmses_lgb)
# print('rmse of xgb:',tr.rmses_xgb) #[0.11601585076934848]
# print('rmse of xgbr:',tr.rmses_xgbr) #[0.11567734530963232]
# print('rmse of mlpr:',tr.rmses_mlpr) #[0.39306480765719265]

# weight_lgb = (1 / tr.rmses_lgb[0])
# weight_xgb = (1 / tr.rmses_xgb[0])
# weight_mlpr = (1 / tr.rmses_mlpr[0])

preds_ans1 = pred.preds_lgb[0] * 0.5 + pred.preds_xgbr[0] * 0.5
submission['SalePrice'] = preds_ans1
submission.to_csv(dir_path +'/submit/sample_submit04.csv', index=False)