# 👑 Leaderboard

|  | pearson | Rank |
| --- | --- | --- |
| Public | 0.9279 | 6 |
| Private | 0.9350 | 4 |

# 🗓️ 개발 기간

2023.04.12 ~ 2023.04.20(총 9일)

# 📄 프로젝트 소개

- 두 문장 사이의 의미 유사성을 측정하는 것을 목적으로 한다. 이를 통해 자연어 처리 모델의 성능을 평가하고, 문장 간의 유사성을 분석하는 등의 다양한 응용이 가능할 것으로 예상된다.

# 💽 사용 데이터셋

- Train Data : 9,324개
- Test Data : 1,100개
- Dev Data : 550개
- Label 점수 : 0~5 사이의 실수로 표현
    - 5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함
    - 4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음
    - 3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
    - 2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함
    - 1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음
    - 0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음

| 이름 | 설명 |
| --- | --- |
| id | 문장 고유 ID. 데이터의 이름과 버전, train / dev / test가 적혀있다. |
| source | 문장의 출처로 총 3가지 source가 존재. |
| sentence1 | 문장 쌍의 첫번째 문장. |
| sentence2 | 문장 쌍의 두번째 문장. |
| label | 문장 쌍에 대한 유사도로 0~5점 소수 첫째 자리까지 표시. |
| binary-label | 문장 쌍에 대한 유사도가 2점 이하인 경우 0, 3점 이상인 경우 1로 변환한 binary label. |

# 📋 평가 지표

- **피어슨 상관 계수 PCC(Pearson Correlation Coefficient)** : 두 변수 X와 Y간의 선형 상관 관계를 계량화한 수치
- 정답을 정확하게 예측하는 것보다, 높은 값은 확실히 높게, 낮은 값은 확실히 낮게 전체적인 경향을 잘 예측하는 것이 중요하게 작용

# 👨‍👨‍👧‍👧 멤버 구성 및 역할

| [곽민석](https://github.com/kms7530) | [이인균](https://github.com/lig96) | [임하림](https://github.com/halimx2) | [최휘민](https://github.com/ChoiHwimin) | [황윤기](https://github.com/dbsrlskfdk) |
| --- | --- | --- | --- | --- |
| <img src="https://avatars.githubusercontent.com/u/6489395" width="140px" height="140px" title="Minseok Kwak" /> | <img src="https://avatars.githubusercontent.com/u/126560547" width="140px" height="140px" title="Ingyun Lee" /> | <img src="https://ca.slack-edge.com/T03KVA8PQDC-U04RK3E8L3D-ebbce77c3928-512" width="140px" height="140px" title="Halim Lim" /> | <img src="https://avatars.githubusercontent.com/u/102031218?v=4" width="140px" height="140px" title="ChoiHwimin" /> | <img src="https://avatars.githubusercontent.com/u/4418651?v=4" width="140px" height="140px" title="yungi" /> |
- **곽민석**
    - 모델 리서치, 프로젝트 구조 세분화, 파라미터 튜닝 및 구조 개선
- **이인균**
    - EDA, 전처리, 모델 실험
- **임하림**
    - 모델 리서치, Cross validation(KFold)을 통한 모델 검증, 모델 실험
- **최휘민**
    - 모델 리서치, 모델 실험, 모델 평가, 모델 앙상블
- **황윤기**
    - 모델 리서치 및 설계, 스케쥴러 적용, 하이퍼 파라미터 튜닝, Wandb 환경 구성

# ⚒️ 기능 및 사용 모델

- `klue/roberta-large`
- `beomi/KcELECTRA`

# 🏗️ 프로젝트 구조

```bash
.
├── Readme.md
├── code
│   ├── Halim
│   │   └── train_kfold.ipynb
│   ├── Ingyun_0424
│   │   └── kcelectra_linearscheduler_totalversion_IfNotTotalTestpEqualZeroDot93.py
│   ├── Minseok
│   │   ├── base_2.py
│   │   ├── dataset.ipynb
│   │   ├── main.py
│   │   └── run.ipynb
│   ├── base_model
│   │   ├── base_2-kobert.py
│   │   ├── base_2.py
│   │   ├── base_2_no_sweep.py
│   │   ├── main.py
│   │   ├── run-kobert.ipynb
│   │   └── run.ipynb
│   ├── inference.py
│   └── train.py
└── data
    ├── dev.csv
    ├── sample_submission.csv
    ├── test.csv
    └── train.csv
```

# 🔗 링크

- [Wrap-up report](/assets/docs/NLP_04_Wrap-Up_Report_STS.pdf)
