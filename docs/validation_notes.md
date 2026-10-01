# Validation notes

## 왜 일반 random split을 사용하지 않았는가

원본 센서 데이터는 약 0.1초 간격으로 연속 측정됩니다. 인접 sample을 임의로 train/test에 나누면 같은 측정 구간의 매우 유사한 window가 양쪽에 포함될 수 있습니다. 따라서 시간 gap이 0.11초를 초과할 때 새로운 `segment_id`를 생성하고, segment를 group으로 묶어 `StratifiedGroupKFold`를 사용했습니다.

## 개발 성능과 최종 성능의 구분

Feature 선택, 모델 비교, aggregation rule 검토, 소규모 hyperparameter tuning을 반복 Group CV 결과를 보면서 수행했습니다. 따라서 저장소의 성능은 **개발 단계의 반복 교차검증 결과**이며 완전히 독립된 최종 test set 성능이 아닙니다.

## 데이터 한계

- 10-sample window 기준 usable abnormal segment: 17개
- abnormal event 수가 작아 segment-level 성능은 한두 segment에 민감함
- 정상/이상 데이터가 서로 다른 수집 세션이라면 session confounding 가능성 존재
- 현재 label만으로 실제 고장 시작시점, 고장 유형, RUL을 확정할 수 없음

## 대표 오류 사례

- Segment 16: segment 내부에서 window 이상확률이 정상형 → 경계형 → 강한 이상으로 변화
- Segment 20: 저진동·고전류 형태의 희귀 패턴으로, 학습 split 구성에 따라 예측확률 변동이 큼

이 사례들은 모델 평균점수만으로 놓칠 수 있는 failure mode를 확인하기 위해 사용했습니다.
