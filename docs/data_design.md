# Data design decisions

## 1. Segmentation threshold: 0.11 s

원본 `TimeStamp` 차이를 직접 확인한 결과, 연속 측정 구간은 사실상 0.1초에 밀집되어 있었습니다.

| Dataset | 연속 간격 0.1 s | 0.11 s 이하 | 0.11 s 초과 | 0.11 s 초과 중 최소 gap |
|---|---:|---:|---:|---:|
| Normal | 19,400 | 19,401 | 598 | 1.545 s |
| Abnormal | 579 | 579 | 20 | 1.780 s |

즉, 실제 데이터에서는 0.11초 근처에 애매한 gap이 존재하지 않았고, 연속 측정 간격 0.1초 다음에는 최소 1.545초 이상으로 크게 벌어졌습니다.

따라서 `0.11 s`는 물리적으로 특별한 값이라기보다, **0.1초 연속 샘플과 명확한 수집 공백을 분리하기 위한 tolerance가 포함된 경계값**입니다.

이 프로젝트에서는 다음처럼 정의했습니다.

```python
new_segment = time_diff.isna() | (time_diff > 0.11)
```

## 2. Window size: 10 samples ≈ 1 s

20-sample(약 2초) window와 비교했을 때 10-sample window가 abnormal 구간을 더 많이 보존했습니다.

| Dataset | Window | usable segments | usable segment ratio | used samples | used sample ratio |
|---|---:|---:|---:|---:|---:|
| Abnormal | 10 samples | 17 / 21 | 80.95% | 530 / 600 | 88.33% |
| Abnormal | 20 samples | 13 / 21 | 61.90% | 420 / 600 | 70.00% |
| Normal | 10 samples | 530 / 599 | 88.48% | 18,200 / 20,000 | 91.00% |
| Normal | 20 samples | 452 / 599 | 75.46% | 14,560 / 20,000 | 72.80% |

특히 20-sample window에서는 길이가 짧은 abnormal segment가 더 많이 제외됩니다. 따라서 현재 데이터에서는 **모델 점수만이 아니라 이상 segment 보존율까지 고려해 10-sample window를 baseline으로 선택**했습니다.

## 3. Window-size sensitivity experiment

동일한 15개 feature, tuned CatBoost, StratifiedGroupKFold 조건에서 0.5초 / 1초 / 2초 window를 비교했습니다.

| Window | Abnormal usable segments | Precision | Recall | F1 | F1 std | Balanced Acc. | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 18 / 21 | 0.964 | 0.886 | 0.922 | 0.028 | 0.943 | **0.982** |
| **1.0 s** | **17 / 21** | **0.982** | **0.905** | **0.941** | 0.043 | **0.952** | 0.976 |
| 2.0 s | 13 / 21 | 1.000 | 0.860 | 0.921 | 0.074 | 0.930 | **0.990** |

### 해석

- **1초 window가 F1, Recall, Balanced Accuracy에서 가장 좋았습니다.**
- 0.5초 window는 더 많은 abnormal segment를 보존하고 PR-AUC도 높았지만, Recall과 F1은 1초보다 낮았습니다.
- 2초 window는 precision과 PR-AUC는 높았지만, usable abnormal segment가 13개로 줄고 Recall과 F1이 낮아졌으며 F1 변동성도 가장 컸습니다.

따라서 현재 데이터에서는 **1초 window가 이상 구간 보존과 분류 성능 사이의 균형이 가장 좋았다**고 판단했습니다.

다만 이 비교도 동일 개발 데이터의 Group CV를 사용한 개발 단계 민감도 분석이므로, 1초가 다른 수집 세션이나 실제 설비 환경에서도 최적이라고 일반화하지 않습니다.

원본 결과는 [`results/window_sensitivity_summary.csv`](../results/window_sensitivity_summary.csv)에 저장했습니다.
