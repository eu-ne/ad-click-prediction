# DACON 광고 클릭 예측

토스 NEXT ML CHALLENGE: 광고 클릭 예측(CTR) 모델 개발

## 프로젝트 개요

토스 앱 광고 노출 데이터를 바탕으로 사용자가 광고를 클릭할 확률을 예측하는 프로젝트입니다.
광고가 노출된 시간, 사용자 정보, 광고 관련 피처, 과거 광고 인기도, 유저 로그 시퀀스 등을 활용해 테스트 데이터의 `clicked` 값을 확률로 예측했습니다.

---

## 데이터 설명

### train.parquet

* 샘플 수: 10,704,179개
* 컬럼 수: 119개
* Target: `clicked`

### test.parquet

* 샘플 수: 1,527,298개
* 컬럼 수: 119개

### 주요 변수

| 변수명            | 설명           |
| -------------- | ------------ |
| `ID`           | 샘플 식별자       |
| `gender`       | 성별           |
| `age_group`    | 연령 그룹        |
| `inventory_id` | 광고 지면 ID     |
| `day_of_week`  | 요일 관련 정보     |
| `hour`         | 광고 노출 시간     |
| `seq`          | 유저 서버 로그 시퀀스 |
| `clicked`      | 광고 클릭 여부     |

### Feature 그룹

| Feature 그룹            | 설명              |
| --------------------- | --------------- |
| `l_feat_*`            | 광고 속성 관련 피처     |
| `feat_a_* ~ feat_e_*` | 광고 정보 영역별 피처    |
| `history_a_*`         | 과거 광고 인기도 관련 피처 |

---

## 분석 과정

### 1. 분석 환경 설정

먼저 필요한 라이브러리를 불러오고, 실험 결과가 최대한 동일하게 나오도록 seed를 고정했습니다.
`random`, `numpy`, `torch`의 seed를 맞췄고, GPU를 사용할 수 있는 경우 CUDA 환경에서 학습하도록 설정했습니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

데이터 크기가 큰 편이라 CPU보다는 GPU를 사용하는 것이 학습 시간 측면에서 더 적합하다고 판단했습니다.

---

### 2. 데이터 로드

학습 데이터와 테스트 데이터는 parquet 형식이었기 때문에 `pandas.read_parquet()`으로 불러왔습니다.
데이터를 불러온 뒤에는 shape를 출력해서 행과 컬럼 수가 맞게 로드되었는지 확인했습니다.

```python
train = pd.read_parquet("./train.parquet", engine="pyarrow")
test = pd.read_parquet("./test.parquet", engine="pyarrow")
```

이번 프로젝트의 목표는 테스트 데이터에 대해 광고 클릭 확률을 예측하는 것이므로, 학습 데이터의 `clicked` 컬럼을 target으로 사용했습니다.

---

### 3. 피처 분리

모델 학습에 바로 사용하지 않을 컬럼을 먼저 제외했습니다.
`clicked`는 예측해야 하는 target이고, `ID`는 식별자이기 때문에 입력 피처에서 제외했습니다.
`seq`는 일반 수치형 피처가 아니라 시퀀스 데이터로 따로 처리하기 위해 제외했습니다.

```python
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]
```

그 다음 범주형 변수와 수치형 변수를 나눴습니다.

```python
cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
```

범주형 변수는 임베딩 레이어에 넣기 위해 따로 처리했고, 나머지 변수들은 수치형 변수로 사용했습니다.

---

### 4. 범주형 변수 처리

범주형 변수는 그대로 모델에 넣을 수 없기 때문에 `LabelEncoder`를 사용해 숫자로 변환했습니다.
이때 train과 test를 따로 인코딩하면 test에만 있는 값 때문에 문제가 생길 수 있어서, 두 데이터를 합친 기준으로 인코딩했습니다.

```python
all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
le.fit(all_values)
```

결측치는 `"UNK"`로 바꿔 하나의 범주처럼 처리했습니다.

---

### 5. Dataset 구성

PyTorch로 학습하기 위해 `ClickDataset` 클래스를 만들었습니다.
하나의 데이터가 들어오면 수치형 피처, 범주형 피처, `seq` 데이터, target 값을 반환하도록 구성했습니다.

`seq` 컬럼은 문자열 형태로 저장되어 있어서, `np.fromstring()`을 사용해 숫자 배열로 바꿨습니다.

```python
arr = np.fromstring(s, sep=",", dtype=np.float32)
```

테스트 데이터에는 target이 없기 때문에, `has_target` 옵션을 둬서 학습 데이터와 테스트 데이터를 같은 Dataset 구조에서 처리할 수 있도록 했습니다.

---

### 6. 시퀀스 데이터 패딩

`seq` 데이터는 샘플마다 길이가 달라서 그대로 batch로 묶을 수 없었습니다.
그래서 `collate_fn`을 따로 정의하고, `pad_sequence()`를 사용해 batch 안에서 길이를 맞췄습니다.

```python
seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
```

학습할 때는 target도 함께 반환해야 하고, 추론할 때는 target이 없기 때문에 학습용과 추론용 collate 함수를 나눠서 작성했습니다.

---

### 7. 모델 구성

모델은 수치형 피처, 범주형 피처, 시퀀스 피처를 함께 사용하는 구조로 만들었습니다.

수치형 피처는 Batch Normalization을 적용했고, 범주형 피처는 Embedding Layer를 사용해 벡터로 변환했습니다.
`seq` 데이터는 시간적 흐름이 있는 로그 데이터라고 보고, LSTM을 사용해 처리했습니다.

```python
self.lstm = nn.LSTM(
    input_size=1,
    hidden_size=lstm_hidden,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)
```

LSTM은 양방향으로 설정했습니다. 이후 LSTM의 hidden state를 수치형 피처, 범주형 임베딩 결과와 합쳐 최종 예측에 사용했습니다.

---

### 8. Cross Network 사용

CTR 예측에서는 광고 지면, 시간, 사용자 정보, 광고 속성 등이 서로 조합되면서 클릭 여부에 영향을 줄 수 있다고 생각했습니다.
그래서 단순히 MLP만 사용하는 대신, 피처 간 상호작용을 반영하기 위해 Cross Network를 추가했습니다.

```python
x = x0 * w(x) + x
```

이 과정을 통해 여러 피처가 함께 작용하는 관계를 어느 정도 학습할 수 있도록 했습니다.

---

### 9. 최종 예측 구조

Cross Network를 거친 값은 MLP에 입력했습니다.
MLP는 512, 256, 128 크기의 hidden layer로 구성했고, 각 layer에 ReLU와 Dropout을 적용했습니다.

```python
hidden_units=[512, 256, 128]
dropout=[0.1, 0.2, 0.3]
```

최종 출력은 클릭 여부에 대한 logit 값입니다.
학습할 때는 `BCEWithLogitsLoss`를 사용했고, 추론할 때는 `sigmoid`를 적용해 클릭 확률로 변환했습니다.

---

### 10. 클래스 불균형 처리

광고 클릭 데이터는 클릭하지 않은 경우가 클릭한 경우보다 훨씬 많을 가능성이 높다고 판단했습니다.
그래서 positive class에 가중치를 주기 위해 `pos_weight`를 계산하고 loss에 반영했습니다.

```python
pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

이렇게 해서 클릭한 샘플이 적더라도 모델이 해당 클래스를 어느 정도 더 중요하게 학습하도록 했습니다.

---

### 11. 모델 학습

모델은 `AdamW` optimizer를 사용해 학습했습니다.
batch size는 1024, epoch은 5, learning rate는 1e-3으로 설정했습니다.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
```

학습률은 `CosineAnnealingWarmRestarts` 스케줄러를 사용해 조정했습니다.
각 epoch마다 train loss를 출력해 학습이 진행되는지 확인했고, GPU를 사용하는 경우 메모리 사용량도 함께 확인했습니다.

---

### 12. 추론 및 제출 파일 생성

학습이 끝난 뒤 모델을 evaluation mode로 바꾸고, 테스트 데이터에 대해 예측을 수행했습니다.
모델 출력값은 logit이기 때문에 `sigmoid`를 적용해 0과 1 사이의 클릭 확률로 변환했습니다.

```python
test_preds = torch.sigmoid(model(num_x, cat_x, seqs, lens)).cpu()
```

이후 `sample_submission.csv` 파일의 `clicked` 컬럼에 예측값을 넣고, 최종 제출 파일을 저장했습니다.

```python
submit = pd.read_csv('./sample_submission.csv')
submit['clicked'] = test_preds
submit.to_csv('./test.csv', index=False)
```

---

## 모델 요약

이번 프로젝트에서는 광고 클릭 확률을 예측하기 위해 PyTorch 기반 CTR 예측 모델을 구현했습니다.
수치형 피처, 범주형 피처, 유저 로그 시퀀스를 각각 다르게 처리한 뒤 하나로 결합해 예측에 사용했습니다.

| 구성 요소   | 적용 방식                      |
| ------- | -------------------------- |
| 수치형 피처  | Batch Normalization        |
| 범주형 피처  | Label Encoding 후 Embedding |
| 시퀀스 피처  | Bidirectional LSTM         |
| 피처 상호작용 | Cross Network              |
| 최종 예측   | MLP                        |
| 손실 함수   | BCEWithLogitsLoss          |
| 불균형 처리  | pos_weight 적용              |
| 제출값     | sigmoid를 적용한 클릭 확률         |

단순히 모든 컬럼을 한 번에 넣는 방식이 아니라, 변수의 성격에 따라 수치형, 범주형, 시퀀스형으로 나누어 처리한 점에 중점을 두었습니다.
