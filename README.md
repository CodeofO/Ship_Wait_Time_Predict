# 🛳️ 항만 內 선박 대기 시간 예측을 위한 선박 항차 데이터 분석 AI 알고리즘 개발
      
|구분|설명|
|------|---|
|프로젝트 명|항만 內 선박 대기 시간 예측을 위한 선박 항차 데이터 분석 AI 알고리즘 개발|
|제안배경|연료 절감 및 온실감스 감축효과를 위해 주최된 경진대회입니다.|
|기간|2023.09 ~ 2023.10|
|주최|HD한국조션해양 AI Center|
|요약|선박의 대기시간(연속, 수치형 변수)를 예측하는 회귀문제를 머신러닝 알고리즘 기반으로 해결했습니다.|
|역할|리더/ 역할 분배/ 도메인 조사/ EDA / 알고리즘 개발|
|팀원|Hong Jae Min, Lee Jae Min|


## 1. 주요 라이브러리(파이썬)
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
XGBoost, Seaborn, Matplotlib


## 2. 수행내용
-조선, 해양 도메인 조사
-변수별 상관관계, 분포 시각화 > 인사이트 획득
-모델링, 최적화
-프로젝트 관리: 일정 및 계획 수립, 진도 관리

## 3. 핵심 아이디어
#### 1) 인사이트
EDA를 통해 DIST변수가 0.1~75 / 75~175 / 175~ 와 같이 3구간으로 나눠지는 것을 확인함
(DIST가 0인 경우는 Target값이 대부분 0인것을 확인했기 때문에 따로 학습)
![image](https://github.com/CodeofO/Ship_Wait_Time_Predict/assets/99871109/dfa70bba-4f1c-489c-9fa9-40977bb4d5fe)



#### 2) 

#### 3) 



순위 : 91 → 34등
성능 : 5.8%% 향상

---

### 결과
