# Data

원본 데이터는 Git 저장소에 포함하지 않습니다.

- 출처: KAMP AI 제조 데이터셋 — 소성가공 예지보전 AI 데이터셋
- 상세 페이지: https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=%EC%86%8C%EC%84%B1%EA%B0%80%EA%B3%B5&page=1&DATASET_SEQ=48&DISPLAY_MODE_SEL=CARD&EQUIP_SEL=&GUBUN_SEL=&FILE_TYPE_SEL=&WDATE_SEL=

아래 두 파일을 이 폴더에 배치합니다.

```text
data/
├── press_data_normal.csv
└── outlier_data.csv
```

`src/train.py`는 위 파일명을 기준으로 데이터를 로드합니다.
