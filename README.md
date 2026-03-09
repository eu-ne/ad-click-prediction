# DACON 광고 클릭 예측 
(토스 NEXT ML CHALLENGE : 광고 클릭 예측(CTR) 모델 개발)

## 프로젝트 개요
토스 앱 광고 노출 데이터를 기반으로
사용자가 광고를 클릭할 확률(CTR)을 예측하는 머신러닝 모델을 개발하는 프로젝트

## 데이터 설명
### train.parquet
1) 샘플 및 컬럼수
- 샘플 수: 10,704,179
- 컬럼 수: 119('ID' 식별자 컬럼 포함)
- Target: clicked

2) 주 변수
- ID : 샘플 식별자
- gender : 성별
- age_group : 연령 그룹
- inventory_id : 지면 ID
- day_of_week : 주번호
- hour : 시간
- seq : 유저 서버 로그 시퀀스

3) feature 그룹
- l_feat_* : 광고 속성  피처 (l_feat_14는 Ads set)
- feat_a_* ~ feat_e_* : 광고 정보 영역 피처(정보영역 a~e)
- history_a_* : 과거 광고 인기도 피처

4) target
- clicked : 광고 클릭 여부 (1/0)
----

### test.parquet
1) 샘플 및 컬럼수
- 샘플 수 : 1,527,298
- 컬럼 수 : 119('ID' 식별자 컬럼 포함)

### sample_submission.csv 제출 파일
1) 예측 대상
- ID : 샘플 식별자
- clicked : 클릭 확률(예측값)

---

## 분석 과정
