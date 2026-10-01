# SHAP error analysis — Abnormal Segment 20

이 문서는 원래 분석 notebook에서 수행한 **18-feature CatBoost baseline / Fold 3**의 Segment 20 오류분석을 보존합니다.

> 주의: 최종 개발 후보는 peak-to-peak 3개를 제거한 15-feature tuned CatBoost입니다. 아래 SHAP은 최종 모델의 global importance가 아니라, **초기 baseline이 Segment 20을 왜 False Negative로 판단했는지 설명하기 위한 local analysis**입니다.

## Target window

- dataset: Abnormal
- segment_id: 20
- window_id: 1872
- actual label: 1
- baseline CatBoost probability: 0.346234
- predicted label at threshold 0.5: 0

주요 raw feature:
- `AI2_Current_mean = 198.483489`
- `AI2_Current_rms = 207.236039`
- `AI2_Current_max = 301.599540`
- `AI0_Vibration_rms = 0.032128`

## Local SHAP interpretation

이 window에서는 전류 관련 feature가 이상 방향으로 강하게 기여했습니다.

- `AI2_Current_min`: +2.273
- `AI2_Current_mean`: +1.189
- `AI2_Current_max`: +0.542

반대로 낮은 진동 특성은 정상 방향으로 기여했습니다.

- `AI0_Vibration_rms`: -1.052
- `AI0_Vibration_mean`: -0.511
- `AI0_Vibration_min`: -0.453

즉 baseline은 **높은 전류 패턴을 이상 신호로 인식하면서도, 낮은 진동 패턴을 정상 신호로 동시에 인식**했고 최종 확률이 0.5 아래로 내려갔습니다.

이 결과를 물리적 고장 원인으로 해석하지 않습니다. SHAP은 해당 모델의 예측 기여도를 설명하며, 실제 고장 메커니즘을 증명하지 않습니다.

전체 값은 [`results/shap_segment20_baseline18.csv`](../results/shap_segment20_baseline18.csv)에 저장했습니다.
