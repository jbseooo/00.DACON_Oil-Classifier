# -DACON-_Oil-Classifier

## 건설기계 오일 상태 분류 AI 경진대회

### (Topic) - 건설장비에서 작동오일의 상태를 실시간으로 모니터링하기 위한 오일 상태 판단 모델 개발 (정상, 이상의 이진분류)

### (MODEL) - SMOTE + RANDOMFOREST + GRID SEARCH
기본적인 이상 상태 분류의 문제이기때문에, Target 불균형이 존재함, 이에 따로 불균형데이터를 처리하지않으면 정확도는 높을 수 있으나 recall(재현율)이 떨어지는 편향된 결과가 나올 있어, 오버샘플링을 통해 클래스 불균형 문제를 해소함
모델링 결과 가장 성능이 좋은 RANDOMFOREST 모델을 사용하였으며, 그리드서치를 통해 최적의 파라미터를 설정함

### (RESULT) - 87 / 1183 (Macro F1 Score : 0.5652)
