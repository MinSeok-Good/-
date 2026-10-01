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

## 3. 아직 남아 있는 한계

0.5초(5 samples), 1초(10 samples), 2초(20 samples)를 동일한 Group CV 조건에서 모두 비교한 성능 민감도 실험은 아직 수행하지 않았습니다. 따라서 1초가 보편적으로 최적이라고 주장하지 않습니다.

추가 검증 시에는 window size별로:
- usable abnormal segment 수
- recall / F1 / PR-AUC
- 오류 segment 변화
를 함께 비교하는 것이 적절합니다.
