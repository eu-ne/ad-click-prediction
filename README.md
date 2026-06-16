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
## 분석 과정

# 1. 분석 환경 설정 및 재현성 확보

프로젝트에 필요한 라이브러리를 불러온 뒤, 실험 결과의 재현성을 확보하기 위해 시드 고정 함수를 정의했다.
`random`, `numpy`, `torch`의 seed를 동일하게 설정하고, CUDA 환경에서도 동일한 결과가 나오도록 `torch.backends.cudnn.deterministic` 옵션을 적용했다.

또한 GPU 사용 가능 여부를 확인하여, CUDA를 사용할 수 있는 경우 GPU 기반으로 모델 학습과 추론을 수행하도록 설정했다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

# 2. 데이터 로드

학습 데이터와 테스트 데이터는 `parquet` 형식으로 제공되어 `pandas.read_parquet()`을 사용해 불러왔다.
데이터 로드 후 `train`, `test`의 shape를 출력하여 데이터가 정상적으로 불러와졌는지 확인했다.

```python
train = pd.read_parquet("./train.parquet", engine="pyarrow")
test = pd.read_parquet("./test.parquet", engine="pyarrow")
```

본 프로젝트의 예측 대상은 광고 클릭 여부를 나타내는 `clicked`이며, 테스트 데이터에 대해 각 사용자가 광고를 클릭할 확률을 예측하는 것이 목표이다.

---

# 3. 피처 분리

모델 학습에 사용하지 않을 컬럼을 제외하고 입력 피처를 구성했다.
예측 대상인 `clicked`, 시퀀스 로그 컬럼인 `seq`, 식별자인 `ID`는 일반 피처에서 제외했다.

```python
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]
```

이후 피처를 범주형 변수와 수치형 변수로 구분했다.

* 범주형 변수: `gender`, `age_group`, `inventory_id`, `l_feat_14`
* 수치형 변수: 범주형 변수를 제외한 나머지 피처

범주형 변수는 임베딩 레이어에 입력되도록 별도로 처리했고, 수치형 변수는 결측치를 0으로 대체한 뒤 모델에 입력했다.

---

# 4. 범주형 변수 인코딩

범주형 변수는 `LabelEncoder`를 사용해 정수형 값으로 변환했다.
학습 데이터와 테스트 데이터에 존재하는 범주를 함께 기준으로 인코딩하기 위해 두 데이터를 합친 뒤 encoder를 학습시켰다.

```python
all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
le.fit(all_values)
```

이렇게 처리한 이유는 테스트 데이터에만 존재하는 범주가 있을 경우 인코딩 오류가 발생하는 것을 방지하기 위해서이다.
결측치는 문자열 `"UNK"`로 대체하여 하나의 범주로 처리했다.

---

# 5. Dataset 구성

PyTorch 학습을 위해 `ClickDataset` 클래스를 정의했다.
해당 클래스는 하나의 샘플에 대해 다음 값을 반환하도록 구성했다.

* 수치형 피처
* 범주형 피처
* 사용자 로그 시퀀스 `seq`
* 타깃값 `clicked`

`seq` 컬럼은 문자열 형태의 시퀀스 데이터이므로, `np.fromstring()`을 사용해 숫자 배열로 변환했다.
타깃값이 존재하는 학습 데이터와 타깃값이 없는 테스트 데이터를 모두 처리할 수 있도록 `has_target` 옵션을 두었다.

---

# 6. 시퀀스 데이터 패딩 처리

`seq`는 샘플마다 길이가 다를 수 있기 때문에, batch 단위 학습을 위해 별도의 `collate_fn`을 정의했다.
`pad_sequence()`를 사용해 batch 내 시퀀스 길이를 맞추고, 실제 시퀀스 길이는 `seq_lengths`로 따로 저장했다.

```python
seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
```

학습용 데이터와 추론용 데이터의 반환 구조가 다르기 때문에 `collate_fn_train`, `collate_fn_infer`를 각각 정의했다.

---

# 7. 모델 구조

CTR 예측을 위해 PyTorch 기반의 `WideDeepCTR` 모델을 구성했다.
모델은 크게 세 가지 입력을 함께 사용한다.

1. 수치형 피처
2. 범주형 피처
3. 사용자 로그 시퀀스 피처

수치형 피처는 `BatchNorm1d`를 적용해 정규화했다.
범주형 피처는 각각 `Embedding Layer`를 거쳐 dense vector로 변환했다.
`seq` 데이터는 LSTM에 입력하여 시퀀스 패턴을 반영했다.

```python
self.lstm = nn.LSTM(
    input_size=1,
    hidden_size=lstm_hidden,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)
```

LSTM은 양방향 구조로 설정하여 시퀀스의 앞뒤 흐름을 모두 반영할 수 있도록 했다.
마지막 hidden state를 사용해 시퀀스 정보를 하나의 벡터로 요약했다.

---

# 8. Cross Network 적용

수치형 피처, 범주형 임베딩, LSTM 출력값을 하나로 결합한 뒤, `CrossNetwork`를 적용했다.
Cross Network는 피처 간 명시적인 교차 관계를 학습하기 위한 구조로, 광고 클릭 예측처럼 여러 피처 간 상호작용이 중요한 문제에 적합하다고 판단했다.

```python
x = x0 * w(x) + x
```

이를 통해 단순한 MLP보다 피처 간 조합 효과를 더 잘 반영할 수 있도록 했다.

---

# 9. MLP 기반 최종 예측

Cross Network를 통과한 피처는 다층 퍼셉트론 구조의 MLP에 입력된다.
MLP는 `[512, 256, 128]` 크기의 hidden layer로 구성했으며, 각 layer마다 `ReLU` 활성화 함수와 `Dropout`을 적용했다.

최종 출력층은 1개의 logit 값을 반환하며, 학습 시에는 `BCEWithLogitsLoss`를 사용했다.
추론 단계에서는 이 logit 값에 `sigmoid`를 적용하여 클릭 확률로 변환했다.

---

# 10. 클래스 불균형 처리

광고 클릭 데이터는 일반적으로 클릭하지 않은 샘플이 훨씬 많은 불균형 데이터이다.
이를 고려해 positive class에 더 큰 가중치를 주기 위해 `pos_weight`를 계산하고, `BCEWithLogitsLoss`에 적용했다.

```python
pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

이를 통해 클릭 샘플이 적더라도 모델이 positive class를 더 잘 학습하도록 했다.

---

# 11. 모델 학습

모델은 `AdamW` optimizer를 사용해 학습했다.
학습률은 `1e-3`, batch size는 `1024`, epoch 수는 `5`로 설정했다.

학습률 스케줄러로는 `CosineAnnealingWarmRestarts`를 사용해 학습 과정에서 learning rate가 주기적으로 조정되도록 했다.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
```

각 epoch마다 train loss를 출력하여 학습 진행 상황을 확인했으며, CUDA 사용 시 GPU 메모리 사용량도 함께 출력했다.

---

# 12. 추론 및 제출 파일 생성

학습이 완료된 모델을 평가 모드로 전환한 뒤, 테스트 데이터에 대해 클릭 확률을 예측했다.
모델 출력값은 logit 형태이므로 `sigmoid` 함수를 적용해 0과 1 사이의 확률값으로 변환했다.

```python
test_preds = torch.sigmoid(model(num_x, cat_x, seqs, lens)).cpu()
```

최종 예측값은 `sample_submission.csv`의 `clicked` 컬럼에 저장한 뒤, 제출 파일인 `test.csv`로 저장했다.

```python
submit = pd.read_csv('./sample_submission.csv')
submit['clicked'] = test_preds
submit.to_csv('./test.csv', index=False)
```

---

## 모델 요약

본 프로젝트에서는 광고 클릭 여부를 예측하기 위해 수치형 피처, 범주형 피처, 사용자 로그 시퀀스를 함께 사용하는 PyTorch 기반 CTR 예측 모델을 구현했다.

* 수치형 피처: Batch Normalization 적용
* 범주형 피처: Embedding Layer 적용
* 시퀀스 피처: Bidirectional LSTM 적용
* 피처 교차: Cross Network 적용
* 최종 예측: MLP 기반 binary classification
* 손실 함수: `BCEWithLogitsLoss`
* 클래스 불균형 처리: `pos_weight` 적용
* 제출값: `sigmoid`를 적용한 클릭 확률

이를 통해 단순한 테이블 기반 모델이 아니라, 광고·사용자·로그 정보를 함께 반영하는 CTR 예측 모델을 구성했다.

