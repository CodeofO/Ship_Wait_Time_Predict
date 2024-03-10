# 🛳️ 항만 內 선박 대기 시간 예측을 위한 선박 항차 데이터 분석 AI 알고리즘 개발
      
|구분|설명|
|------|---|
|프로젝트 명|항만 內 선박 대기 시간 예측을 위한 선박 항차 데이터 분석 AI 알고리즘 개발|
|제안배경|연료 절감 및 온실감스 감축효과를 위해 주최된 경진대회입니다.|
|기간|2023.09 ~ 2023.10|
|주최|HD한국조션해양 AI Center, DACON|
|요약|선박의 대기시간(연속, 수치형 변수)를 예측하는 회귀문제를 머신러닝 알고리즘 기반으로 해결했습니다.|
|역할|리더/ 역할 분배/ 도메인 조사/ EDA / 알고리즘 개발|
|팀원|Hong Jae Min, Lee Jae Min|

**경진대회 더 자세히 알아보기 🔻**
https://dacon.io/competitions/official/236158/overview/description
<br>
<br>

## 1. 주요 라이브러리(파이썬)
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
XGBoost, Seaborn, Matplotlib


## 2. 수행내용
-조선, 해양 도메인 조사 <br>
-변수별 상관관계, 분포 시각화 > 인사이트 획득<br>
-모델링, 최적화<br>
-프로젝트 관리: 일정 및 계획 수립, 진도 관리<br>

## 3. 핵심 아이디어

#### 1) 인사이트

EDA를 통해 DIST변수가 0.1~75 / 75~175 / 175~ 와 같이 3구간으로 나눠지는 것을 확인함 <br>
(DIST가 0인 경우는 Target값이 대부분 0인것을 확인했기 때문에 따로 학습) <br>
![image](https://github.com/CodeofO/Ship_Wait_Time_Predict/assets/99871109/dfa70bba-4f1c-489c-9fa9-40977bb4d5fe)

'Container' 을 제외한 나머지 범주는 대부분 DIST가 75이상이 될 때부터 확연하게 줄어드는 것과  <br>
'Container'은 175에서 끊기는 것을 확인할 수 있음<br>
<br>
👉 모델에게 저 상황들을 인지시키면 학습을 더 잘할 수 있을거라는 가설세움


#### 2) 도출된 인사이트를 모델링에 적용

##### (1) 구간화 수행
DIST변수를 EDA기반으로 구간화 후 새로운 변수(DIST_binded)로 저장함 <br>
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

##### (2) 특성 공학(Feature Engineering)

항구의 국가명, 선박의 종류, 구간화된 DIST 변수를 **string Type**으로 변경 후 모두 이어진 새로운 변수를 생성
ex) 'KOR' + 'Container' + '02' = 'KOR_Container_02'

          train_1['SPLIT_CRETERION'] = 
          train_1['ARI_CO'].astype(str) 
          + '-' +train_1['SHIP_TYPE_CATEGORY'].astype(str) # 선박의 종류
          + '-' +train_1['DIST_binded'].astype(str) # DIST



#### 3) Stratified -K FOLD

(2)에서 만든 변수를 Stratified -K FOLD 수행 시 데이터셋 분할 기준으로 설정함

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(train_x, SPLIT_CRETERION)):
        train_set = train_x.iloc[train_index]
        valid_set = train_x.iloc[test_index]

        x_train, y_train = train_set, train_y.iloc[train_index]
        x_val, y_val = valid_set, train_y.iloc[test_index]

        model.fit(x_train, y_train, verbose=True)



**본 아이디어를 적용한 결과(최종 결과 X)** <br>
* 순위 : 91 → 34등<br>
* 성능 : 5.8%% 향상<br>

---

### 결과
상위 9%

