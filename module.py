import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error


'''
NOTE

크게 Preprocessing, Modeling 모듈로 나누어짐

1. Preprocessing(전처리)
    1) feature engineering(1)
    2) bining(1)
    3) feature engineering(2)
    4) feature engineering(3)
    5) feature engineering(4)
    6) bining(2)
    7) drop columns(1)
    8) encoding
    9) drop columns(2)
    10) scaling

2. Modeling
    1) 데이터셋 분할
    2) K-Fold 데이터셋 분할 기준 생성(1)
    3) K-Fold 데이터셋 분할 기준 생성(2)
    4) 변수 중요도 추출 및 시각화, 전처리
    5) 최적 하이퍼파라미터 조합
    6) Stratified K-Fold 검증

각 모듈별 설명은 주석으로 코드 위에 작성
    
'''


# NOTE : Preprocessing 1 / feature engineering
# datetime > year, month, day, hour, minute, weekday
def pre_1_datetime(data):
    data['ATA'] = pd.to_datetime(data['ATA'])

    # datetime을 여러 파생 변수로 변환
    data['year'] = data['ATA'].dt.year
    data['month'] = data['ATA'].dt.month
    data['day'] = data['ATA'].dt.day
    data['hour'] = data['ATA'].dt.hour
    data['minute'] = data['ATA'].dt.minute
    data['weekday'] = data['ATA'].dt.weekday
    return data





# NOTE : Preprocessing 2 / bining
# 요일 -> 주중(0) / 주말(1)로 변경
# weekday는 삭제
# DIST == 0 일 경우에만 적용

def pre_2_weekday(data):

    weekday_binded = list()

    for day in data['weekday']:
        if day < 5:
            weekday_binded.append(0)
        else:
            weekday_binded.append(1)

    data['weekday_binded'] = weekday_binded

    return data





# NOTE : Preprocessing 3 / feature engineering
# CN(중국)과 TW(타이완)에 동일한 이름을 가진 항구명 존재 : EKP8
# EKP8_CN, EKP8_TW로 분리시킴
def pre_3_ARI_PORT(data):
    
    cond_EKP8 = data['ARI_PO'] == 'EKP8'
    cond_CN = data['ARI_CO'] == 'CN'
    cond_TW = data['ARI_CO'] == 'TW'

    data.loc[(cond_EKP8 & cond_CN), 'ARI_PO'] = 'EKP8_CN'
    data.loc[(cond_EKP8 & cond_TW), 'ARI_PO'] = 'EKP8_TW'

    return data





# NOTE : Preprocessing 4 / feature engineering
# 선박 전체 깊이 중 물에 잠겨있는 깊이의 비율 : 흘수 높이 / 선박의 깊이
# DRAUGHT : 흘수 높이(선체가 물에 잠긴 깊이)
# DEPTH : 선박의 깊이
def pre_4_ratio_under_water(data):

    data['ratio_under_water'] = 0.0

    cond = (data['DRAUGHT'] != 0) & (data['DEPTH'] != 0)
    data.loc[cond, 'ratio_under_water'] = np.round(data.loc[cond, 'DRAUGHT'] / data.loc[cond, 'DEPTH'], 2)
    data.loc[~cond, 'ratio_under_water'] = np.round((data.loc[~cond, 'DRAUGHT']+1) / (data.loc[~cond, 'DEPTH']+1), 2)
    
    return data





# NOTE : Preprocessing 5 / feature engineering
# 선박의 전체 재화중량톤수를 물에 잠겨있는 깊이의 비율만큼 곱한 수치 : (흘수 높이 / 선박의 깊이) * 재화중량톤수
# DRAUGHT : 흘수 높이(선체가 물에 잠긴 깊이)
# DEPTH : 선박의 깊이
# DEADWEIGHT : 선박의 재화중량톤수(선박이 실을 수 있는 화물의 최대 중량을 톤으로 나타낸 수치)
def pre_5_ratio_under_water_DEADWEIGHT(data):

    data['ratio_under_water_DEAD'] = 0.0

    cond = (data['DRAUGHT'] != 0) & (data['DEPTH'] != 0)
    data.loc[cond, 'ratio_under_water_DEAD'] = np.round(data.loc[cond, 'DRAUGHT'] / data.loc[cond, 'DEPTH'], 2) * data.loc[cond, 'DEADWEIGHT']
    data.loc[~cond, 'ratio_under_water_DEAD'] = np.round((data.loc[~cond, 'DRAUGHT']+1) / (data.loc[~cond, 'DEPTH']+1), 2) * data.loc[~cond, 'DEADWEIGHT']
    
    return data





# NOTE : Preprocessing 6 / 변수 구간화
# EDA기반 얻은 인사이트를 토대로 구간화된 DIST
#  0~75 / 75~150 / 150~175 / 175~ 로 구간화 된다.
def pre_6_dist_binding(data):

    dist_l = list()

    for d in data['DIST']:

        if d < 75:
            dist_l.append(str('00'))
        elif d < 150:
            dist_l.append(str('01'))
        elif d < 175:
            dist_l.append(str('02'))
        else:
            dist_l.append(str('03'))
       
    data['DIST_binded'] = dist_l

    return data





# NOTE : Preprocessing 7 / 변수 제거
# drop : 'ID', 'SAMPLE_ID', 'SHIPMANAGER', 'ATA', 'FLAG'
    # feature importance(피쳐중요도) 확인 후 모델 성능에 영향을 거의 미치지 않는 변수들
    # drop 했을 때 성능이 향상되는 변수들
def pre_7_drop(data):
    
    cols_drop = ['ID', 'SAMPLE_ID', 'SHIPMANAGER', 'ATA', 'FLAG']
    data = data.drop(cols_drop, axis=1)
    
    return data




# NOTE : Preprocessing 8 / encoding
# 수치형 변수를 제외한 범주형 변수 대상으로 Label Encoding 수행\
# Label Encoding : 범주형 데이터를 연속되는 숫자로 변환함(0부터 시작)
def pre_8_encoding(train, test):

    # columns 리스트 생성
    cols_total = train.columns.tolist()

    # 수치형 변수 이름만 있는 list
    cols_numeric = train.describe().columns.tolist()

    # 범주형 변수만 리스트(cols_total)에 남기기
    for col in cols_numeric:
        cols_total.remove(col)

    cols_cate = cols_total

    # Labelencoding 수행
    encoder_le = LabelEncoder()

    for col in cols_cate:
        train[col] = encoder_le.fit_transform(train[col])
        test[col] = encoder_le.transform(test[col])

    return train, test





# NOTE : Preprocessing 9 / 변수 제거
# DIST가 0인 데이터셋에서만 한정해서 제거
# 후진 변수선택법을 통한 실험으로 알게됨
def pre_9_drop(data):
    
    cols_drop = ['weekday']
    data = data.drop(cols_drop, axis=1)
    
    return data




# NOTE : Preprocessing 10 / Scaling
# 학습 전 최종 전처리
# X : Standard Scaling 수행
# Y : target(CI_HOUR)변수에 ln(1+x) 변환 수행
def pre_10_scaling(train, test):

    train_copy = train.copy()
    test_copy = test.copy()

    train_y = np.log1p(train_copy['CI_HOUR'])
    train_x = train_copy.drop(['CI_HOUR'], axis=1)

    scaler = StandardScaler()
    train_x[train_x.columns] = scaler.fit_transform(train_x)
    test_copy[test_copy.columns] = scaler.transform(test_copy)

    return train_x, train_y, test_copy





# NOTE : Modeling 1 / 데이터셋 분할
# DIST가 0인 samples, 0이 아닌 samples로 분할
# 분할이유 
    #: EDA결과 DIST가 0인 경우 대부분 target(CI_HOUR)가 0이였음.
    #  따라서 DIST가 0이 아닌 경우는 DIST가 0인 상황을 제외하고 학습하는게 좋다고 판단.
def model_1_split_DIST(train, test):

    train_0 = train[train['DIST'] == 0]
    train_1 = train[train['DIST'] != 0]

    test_0 = test[test['DIST'] == 0]
    test_1 = test[test['DIST'] != 0]

    print(f'DIST = 0(train) : {train_0.shape}')
    print(f'DIST != 0(train) : {train_1.shape}')

    return train_0, train_1, test_0, test_1





# NOTE : Modeling 2 / K-Fold 데이터셋 분할 기준 생성(1)
# train_0, test_0 : target(CI_HOUR)이 24시간 미만이면 0, 이상이면 1
# train_1, test_1 : target(CI_HOUR)이 100시간 미만이면 0, 이상이면 1
# 24시간, 100시간 선정 근거
    # 24시간(DIST=0) : EDA 결과 DIST가 0일때 24시간 넘어가는 sample은 6개가 존재
    # 100시간(DIST!=0) : EDA 결과 CH_HOUR가 100이상이 될 때 Bulk 선박을 제외한 나머지 선박종류의 분포가 0에 수렴함
# 추후 K-Fold 교차 검증시 선박 종류, 항구명과 함께 기준으로 사용

def modeling_2_creterion_KF(train, creterion_num):

    # 데이터프레임 복사
    train_copy = train.copy()

    # 'CI_HOUR_binded' 열 추가
    train_copy['CI_HOUR_binded'] = train_copy['CI_HOUR'].apply(lambda x: '0' if x < creterion_num else '1')

    return train_copy





# NOTE : Modeling 3 / K-Fold 데이터셋 분할 기준 생성(2)
# 항구명, 선박 종류, CI_HOUR_binded(구간화된 target)을 사용해 K-Fold 기준 생성
# train_0(DIST=0)의 기준 : 항구명, 선박 종류, CI_HOUR_binded이 Feature Engineering된 것
# train_1(DIST!=0)의 기준 : 항구명, 선박 종류, CI_HOUR_binded, DIST_binded이 Feature Engineering된 것
def modeling_3_make_creterion(train_0, train_1):
    # make split cond
    train_0['SPLIT_CRETERION'] = train_0['ARI_PO'].astype(str) + '-' +train_0['SHIP_TYPE_CATEGORY'].astype(str)
    SPLIT_CRETERION_0 = train_0['SPLIT_CRETERION']
    train_0.drop(['SPLIT_CRETERION', 'CI_HOUR_binded'], axis=1, inplace=True)

    train_1['SPLIT_CRETERION'] = train_1['ARI_CO'].astype(str) + '-' +train_1['SHIP_TYPE_CATEGORY'].astype(str) + '-' +train_1['DIST_binded'].astype(str)#+ '-' +train_1['CI_HOUR_binded_1'].astype(str) 
    SPLIT_CRETERION_1 = train_1['SPLIT_CRETERION']
    train_1.drop(['SPLIT_CRETERION', 'CI_HOUR_binded'], axis=1, inplace=True)

    return SPLIT_CRETERION_0, SPLIT_CRETERION_1






# NOTE : Modeling 4 / 변수 중요도 추출 및 시각화, 전처리
# threshold보다 중요도가 낮은 column은 제거
# 실험에 의해 threshold = 0.004가 가장 좋은 성능을 보여줌
def modeling_4_feature_importance(train_x, train_y, test, model, model_name, threshold):

    print(f'Model Tune for {model_name}.')
    model.fit(train_x, train_y)

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, len(train_x.columns) // 2.5))
    plt.title(f"Feature Importances ({model_name})")

    # Color distinction for features above vs below threshold
    for i, v in enumerate(feature_importances[sorted_idx]):
        plt.barh(i, v, align='center', color='blue' if v >= threshold else 'red')

    plt.yticks(range(train_x.shape[1]), train_x.columns[sorted_idx])
    plt.xlabel('Importance')
    plt.show()

    low_importance_features = train_x.columns[feature_importances < threshold]

    train_reduced = train_x.drop(columns=low_importance_features)
    test_reduced = test.drop(columns=low_importance_features)

    print('💡 제거된 column들 \n:', low_importance_features.tolist())

    return train_reduced, test_reduced, model, feature_importances





# NOTE : Modeling 5 / 최적 하이퍼파라미터
# Grid Search를 사용해 모델의 최적 하이퍼파라미터 조합발견
def modeling_5_grid_search(train_x, train_y, model, param_grid):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=kf, verbose=0, n_jobs=-1)
    grid_search.fit(train_x, train_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_model, best_params




# NOTE : Modeling 6 / Stratified K-Fold
# best model에 Stratified K-Fold적용 후 학습, 검증, 추론 수행
def modeling_6_train_evaluation(train_x, train_y, test, best_model, SPLIT_CRETERION):

    model = best_model

    scores = []
    ensemble_predictions = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(train_x, SPLIT_CRETERION)):
        train_set = train_x.iloc[train_index]
        valid_set = train_x.iloc[test_index]

        x_train, y_train = train_set, train_y.iloc[train_index]
        x_val, y_val = valid_set, train_y.iloc[test_index]

        model.fit(x_train, y_train, verbose=True)

        # 각 모델로부터 Validation set에 대한 예측을 평균내어 앙상블 예측 생성
        val_pred = model.predict(x_val)


        # Validation set에 대한 대회 평가 산식 계산 후 저장
        # log scaling 되돌리기
        scores.append(mean_absolute_error(np.expm1(y_val), np.expm1(val_pred)))
        
        preds = model.predict(test)
        preds = np.where(preds < 0, 0, preds)

        ensemble_predictions.append(np.expm1(preds))

    # K-fold 모든 예측의 평균을 계산하여 fold별 모델들의 앙상블 예측 생성
    final_predictions = np.mean(ensemble_predictions, axis=0)

    # 각 fold에서의 Validation Metric Score와 전체 평균 Validation Metric Score출력
    print('\n\nPrediction Score')
    print("Validation : MAE for each fold:", scores)
    print("Validation : MAE:", np.mean(scores))

    return final_predictions