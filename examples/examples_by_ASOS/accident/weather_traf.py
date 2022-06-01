
# coding: utf-8

# In[ ]:


# 0. 컬럼명	구분 	명칭 ----
#==============================================================#
# _결과 ----
# Y	결과	사고유무

# _기타 ----
# month	사고 발생 월
# hour	사고 발생 시간

# _교통류 ----
# AVG_SPEED	1시간 평균 속도
# VOLUME_ALL	1시간 총 교통량

# _기상 ----
# TA_MAX_2_STD	이전 2시간 동안 최대 기온의 편차
# WS_MAX_6_STD	이전 6시간 동안 최대 풍속의 편차
# WS_MAX_2_MIN	이전 2시간 동안 최대 풍속의 최소
# TA_MIN_6_MAX	이전 6시간 동안 최소 기온의 최대
# WS_AVG_2_STD	이전 2시간 동안 평균 풍속의 편차
# TA_MIN_6_STD	이전 6시간 동안 최소 기온의 편차
# RN_3_MAX	이전 3시간 동안 강수의 최대

# _기하구조 ----
# GISID	1KM 구간 번호
# CONZONE	콘존ID
# BUSlane	버스전용차로 유무
# JD_3KM_STD	이전 3KM 구간 동안 종단 편차
# PM_3KM_STD	이전 3KM 구간 동안 평면 편차
# JD_1KM_AVG	이전 1KM 구간 동안 종단 평균
# KS_3KM_AVG	이전 3KM 구간 동안 경사 평균
# BY_3KM_STD	이전 3KM 구간 동안 방영 편차
# BY_3KM_AVG	이전 3KM 구간 동안 방영 평균
# SPEEDLIMIT	제한속도


# In[ ]:


# 1. 데이터 로딩 ----
#==============================================================#
# _패키지 로딩 ----
#==============================================================#
import pandas as pd # 데이터 분석을 위한 라이브러리
import matplotlib.pyplot as plt # 시각화를 위한 라이브러리
import seaborn as sns # 시각화를 위한 라이브러리
import numpy as np # 다차원 배열을 위한 라이브러리
import sys # 파이썬 인터프리터를 제어하는 라이브러리

from sklearn.model_selection import train_test_split # 학습 및 테스트 데이터들을 무작위로 분할
from sklearn.ensemble import GradientBoostingRegressor # 모델을 위한 라이브러리
from sklearn.preprocessing import LabelEncoder # 원-핫 인코딩을 위한 라이브러리

#================================================================#
# _분석 환경 설정 ----
#================================================================#

# Python 메모리에 생성된 모든 객체 삭제(초기화)
sys.stdout.flush()

# 경고 메세지를 출력되지 않도록 합니다.
import warnings
warnings.filterwarnings(action="ignore")


#=============================================================
# 작업 디렉터리 경로 확인
#=============================================================
import os
currentPath = os.getcwd() # 현재 위치한 디렉토리 경로 확인
print('Current working dir : %s' % currentPath)


# In[ ]:


# 데이터 로딩
#=============================================================
# 데이터 읽어오기
#=============================================================
# 기상 데이터
data_weather = pd.read_csv("weather.csv")

# 불러온 데이터 구조 확인하기
data_weather.info()


# In[ ]:


#=============================================================
# 도로기하 데이터
#=============================================================
data_road = pd.read_csv("road.csv")

# 불러온 데이터 구조 확인하기
data_road.info()


# In[ ]:


#=============================================================
# 교통류 데이터
#=============================================================
data_traf = pd.read_csv("traf.csv")

# 불러온 데이터 구조 확인하기
data_traf.info()


# In[ ]:


#=============================================================
# 데이터 결합
#=============================================================
# 테이블 결합 및 확인
df_raw = pd.merge(data_weather, data_traf)
weather_traf = pd.merge(df_raw, data_road)
weather_traf.head()


# In[ ]:


# 2. 데이터 탐색
#=============================================================
# 데이터 요약 : 타입확인
weather_traf.info()


# In[ ]:


#=============================================================
# 탐색적 데이터 분석
#=============================================================
# 데이터 기술·통계
weather_traf.describe(include="all")


# In[ ]:


# 결측치 파악
weather_traf.isnull().sum()


# In[ ]:


#=============================================================
# ___시각화
# ___1km 구간에 따른 사고 빈도
# 1km구간을 칭하는 GISID에 대해 각 Y는 SUM(), 각각의 하위 데이터는 Mean()을 하여 시각화
# 색상코드
# 기상: green
# 기하: yellow
# 교통: navy

# ____ 기상
#=============================================================
# 1KM 구간에 따른 사고 빈도 (기상)
#=============================================================
plt.figure(figsize=(26, 24))
for i, col in enumerate(weather_traf.filter(regex="TA|WS|RN")):
    weather_traf_1km = weather_traf.groupby("GISID")[[col, "Y"]].agg({col:"mean", "Y":"sum"})
    plt.subplot(4, 2, i + 1)
    sns.scatterplot(data=weather_traf_1km, x=col, y="Y", color="g")
    plt.title(col + " by count ")


# In[ ]:


# ____ 기하구조
#=============================================================
# 1KM 구간에 따른 사고 빈도 (기하구조)
#=============================================================
plt.figure(figsize=(26, 24))
for i, col in enumerate(weather_traf.filter(regex="JD|PM|KS|BY|LIMIT")):
    weather_traf_1km = weather_traf.groupby("GISID")[[col, "Y"]].agg({col:"mean", "Y":"sum"})
    plt.subplot(4, 2, i + 1)
    sns.scatterplot(data=weather_traf_1km, x=col, y="Y", color="y")
    plt.title(col + " by count ")


# In[ ]:


# ____ 교통류
#=============================================================
# 1KM 구간에 따른 사고 빈도 (교통류)
#=============================================================
plt.figure(figsize=(12, 6))
for i, col in enumerate(weather_traf.filter(regex="_SPEED|VOLUME")):
    weather_traf_1km = weather_traf.groupby("GISID")[[col, "Y"]].agg({col:"mean", "Y":"sum"})
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=weather_traf_1km, x=col, y="Y", color="navy")
    plt.title(col + " by count ")


# In[ ]:


# ___콘존에 따른 사고 빈도
# 콘존을 칭하는 CONZONE에 대해 각 Y는 SUM(), 각각의 하위 데이터는 Mean()을 하여 시각화
# 색상코드
# 기상: green
# 기하: yellow
# 교통: navy

# ____ 기상
#=============================================================
# 콘존에 따른 사고 빈도 (기상)
#=============================================================
plt.figure(figsize=(26, 24))
for i, col in enumerate(weather_traf.filter(regex="TA|WS|RN")):
    weather_traf_con = weather_traf.groupby("CONZONE")[[col, "Y"]].agg({col:"mean", "Y":"sum"})
    plt.subplot(4, 2, i + 1)
    sns.scatterplot(data=weather_traf_con, x=col, y="Y", color="g")
    plt.title(col + " by count ")


# In[ ]:


# ____ 기하구조
#=============================================================
# 콘존에 따른 사고 빈도 (기하구조)
#=============================================================
plt.figure(figsize=(26, 24))
for i, col in enumerate(weather_traf.filter(regex="JD|PM|KS|BY|LIMIT")):
    weather_traf_con = weather_traf.groupby("CONZONE")[[col, "Y"]].agg({col:"mean", "Y":"sum"})
    plt.subplot(4, 2, i + 1)
    sns.scatterplot(data=weather_traf_con, x=col, y="Y", color="y")
    plt.title(col + " by count ")


# In[ ]:


# ____ 교통류
#=============================================================
# 콘존에 따른 사고 빈도 (교통류)
#=============================================================
plt.figure(figsize=(12, 6))
for i, col in enumerate(weather_traf.filter(regex="_SPEED|VOLUME")):
    weather_traf_con = weather_traf.groupby("CONZONE")[[col,"Y"]].agg({col:"mean", "Y":"sum"})
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=weather_traf_con, x=col, y="Y", color="navy")
    plt.title(col + " by count ")


# In[ ]:


#=============================================================
# 히스토그램 (발생건수, 빈도)
#=============================================================
sns.histplot(data=weather_traf, x="Y", bins=15)
plt.title("Frequency by count")
plt.xlabel("Count")
plt.ylabel("Frequency")


# In[ ]:


# 3. 데이터 처리
#=============================================================
# 이상치 제거 전
#=============================================================
print("이상치 제거 전:", weather_traf.shape)

# 이상치 제거
weather_traf = weather_traf.loc[(weather_traf["TA_MAX_2_STD"] < 7) &
                                (weather_traf["WS_MAX_2_MIN"] < 7) &
                                (weather_traf["TA_MIN_6_STD"] < 10) &
                                (weather_traf["RN_3_MAX"] < 20) &
                                (weather_traf["PM_3KM_STD"] < 10000)]

# 이상치 제거 후
print("이상치 제거 후:", weather_traf.shape)


# In[ ]:


# 4. 모형 구축
# 분석 데이터 셋
#=============================================================
# 범주형 데이터를 수치데이터로 변환
#=============================================================
le = LabelEncoder()
weather_traf["CONZONE"] = le.fit_transform(weather_traf["CONZONE"])
weather_traf.columns[weather_traf.dtypes.values == "object"]


# In[ ]:


#=============================================================
# 데이터 셋 분할
#=============================================================
# set x and y
X = weather_traf.drop("Y", axis=1)
y = weather_traf["Y"]

# split dataset
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1234)
X_train.shape, X_test.shape, X_valid.shape


# In[ ]:


# 모형 구축
#============================================================
# 모형 튜닝 자동화
#============================================================
# cartesian grid search
rate = [0.1, 0.05, 0.001]
depth = [12, 15, 18]
result1 = []
result2 = []
result3 = []

for i in rate: # learning_rate를 0.1, 0.05, 0.001 로 변경하면서 결과 확인
    for j in depth: # max_depth를 12, 15, 18 로 변경하면서 결과 확인
        result1.append(i)
        result2.append(j)
        Traffic_GBM_Model = GradientBoostingRegressor(learning_rate=i, max_depth=j,
                                                      subsample=0.85, # col_sample_rate
                                                      min_samples_leaf=0.075, # min_rows: 218245의 0.075=16384.0
                                                      max_features=0.51, # col_sample_rate * col_sample_rate_per_tree = 0.51
                                                      min_impurity_decrease=0.0, # min_split_improvement
                                                      n_estimators=148, # ntrees
                                                      random_state=1234) # seed
        # 분석환경을 고려한 빠른 결과 산출을 위해, 10% 의 데이터만 사용
        Traffic_GBM_Model.fit(X_train.sample(frac=0.1, random_state=1234), y_train.sample(frac=0.1, random_state=1234)) # training_frame = X_train
        pred = Traffic_GBM_Model.predict(X_valid.sample(frac=0.1, random_state=1234))  # validation_frame = X_valid
        result3.append(np.sqrt(np.sum((y_valid.sample(frac=0.1, random_state=1234) - pred) ** 2) / len(pred)))


# In[ ]:


#=============================================================
# RMSE가 낮은 순으로 정렬하기
#=============================================================
gbm_gridperf = pd.DataFrame({"learning_rate" : result1,
                             "max_depth" : result2,
                             "RMSE" : result3}).sort_values(by="RMSE")
gbm_gridperf


# In[ ]:


#============================================================
# 모형 선택
#============================================================
# create Model
Traffic_GBM_Model = GradientBoostingRegressor(max_depth=gbm_gridperf.loc[0]['max_depth'],
                          learning_rate=gbm_gridperf.loc[0]['learning_rate'],
                          subsample=0.85, # col_sample_rate
                          min_samples_leaf=0.075, # min_rows: 218245의 0.075=16384.0
                          max_features=0.51, # col_sample_rate * col_sample_rate_per_tree = 0.51
                          min_impurity_decrease=0.0, # min_split_improvement
                          n_estimators=148, # ntrees
                          random_state=1234) # seed

# __Model Summary ----
Traffic_GBM_Model
Traffic_GBM_Model.fit(X_train, y_train)  # training_frame = train
pred = Traffic_GBM_Model.predict(X_valid)  # validation_frame = valid


# In[ ]:


# __RMSE ----
print("MAE :", np.mean(abs(y_valid - pred)) )
print("MSE :", np.sum((y_valid - pred) ** 2) / len(pred) )
print("RMSE :", np.sqrt(np.sum((y_valid - pred) ** 2) / len(pred)) )
print("RMSLE :", np.sqrt(np.mean(np.power(np.log1p(pred) - np.log1p(y_valid), 2))))


# In[ ]:


# 5. 모형 검증
#=============================================================
# 변수 중요도 파악(표)
#=============================================================
feature_list = pd.DataFrame({"importances":Traffic_GBM_Model.feature_importances_,
                             "feature_names":X.columns.tolist()})
feature_list = feature_list.sort_values("importances", ascending=False)
feature_list["scaled_importance"] = feature_list["importances"] / feature_list["importances"].values[0]
feature_list = feature_list[["feature_names", "scaled_importance", "importances"]].reset_index(drop=True)
feature_list


# In[ ]:


# 변수 중요도 파악(그래프)
plt.figure(figsize=(10, 7))
plt.title("Variable Importance : GBM", fontsize=16)
sns.barplot(data = feature_list, x="importances", y="feature_names", color="b")


# In[ ]:


#============================================================
# 모형 입력변수 선택
#============================================================
i_index = []
rmse = []
Traffic_GBM_Model_feature = Traffic_GBM_Model
for i in [3, 5, 7, 9]: # 3, 5, 7, 9개의 변수 선택 위해 반복문 사용
    cols_to_set = ["CONZONE", "GISID", "month", "hour"]
    cols = cols_to_set + list(feature_list.feature_names[:i])

    train_feature = X_train[cols]
    valid_feature = X_valid[cols]

    Traffic_GBM_Model_feature.fit(train_feature, y_train)  # training_frame = train_feature
    pred = Traffic_GBM_Model_feature.predict(valid_feature)  # validation_frame = valid_feature

    i_index.append(i)
    rmse.append(np.sqrt(np.sum((y_valid - pred) ** 2) / len(pred)))
    print(i, "RMSE :", np.sqrt(np.sum((y_valid - pred) ** 2) / len(pred)))


# In[ ]:


#============================================================
# Model rmse
#============================================================
feature_test = pd.DataFrame({"RMSE":rmse}, index=i_index).sort_values("RMSE", ascending=True)

feature_test


# In[ ]:


#=============================================================
# 최종 모형 선택
#=============================================================
best_i = feature_test.index[0]
cols = cols_to_set + list(feature_list.feature_names[:best_i])

train_feature = X_train[cols] # train_feature <- train
test_feature = X_test[cols]  # test_feature <- test

# create Model
Traffic_GBM_Model_final = Traffic_GBM_Model

# Model Summary
Traffic_GBM_Model_final


# In[ ]:


# 모형 성능 및 예측력 파악
#=============================================================
# 모형 성능
#=============================================================
# Predict on test set
Traffic_GBM_Model_final.fit(train_feature, y_train)
pred = Traffic_GBM_Model_final.predict(test_feature)

print("MODEL : Traffic_GBM_Model_final")
print("MAE :", np.mean(abs(y_test - pred)))
print("MSE :", np.sum((y_test - pred) ** 2) / len(pred))
print("RMSE :", np.sqrt(np.sum((y_test - pred) ** 2) / len(pred)))
print("RMSLE :", np.sqrt(np.mean(np.power(np.log1p(pred) - np.log1p(y_test), 2))))


# In[ ]:


#=============================================================
#  RMSE
#=============================================================
Pred_conversion = pd.DataFrame({"pred_col":pred, "y_test_col":y_test}).set_index("y_test_col")
RMSE = np.sqrt(np.sum((y_test - pred) ** 2) / len(pred))
plt.figure(figsize=(12, 6))
sns.scatterplot(data=Pred_conversion, x=Pred_conversion.index, y="pred_col")
plt.title("rmse = " + str(RMSE), fontsize=16, loc="left")
plt.xlabel("TRUE")
plt.ylabel("Prediction")

