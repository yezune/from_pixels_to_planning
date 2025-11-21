# Experiment 5: Performance Comparison (Flat vs Hierarchical)

## 1. 이론적 배경 (Theoretical Background)

이 실험은 단일 계층(Flat) 모델과 계층적(Hierarchical) 모델의 성능을 정량적으로 비교합니다. 이론적으로 계층적 모델은 다음과 같은 이점을 가져야 합니다:
1.  **Sample Efficiency**: 상위 레벨의 추상화를 통해 더 적은 데이터로 효율적인 정책을 학습.
2.  **Long-term Planning**: 긴 시간 간격(Time Horizon)을 고려한 계획 수립 능력.
3.  **Generalization**: 새로운 상황이나 노이즈에 대한 강건함.

## 2. 실험 설정 (Setup)

*   **비교 대상**:
    *   **Baseline**: Flat Active Inference Agent (Experiment 1 모델).
    *   **Proposed**: Hierarchical Active Inference Agent (Experiment 2 모델).
*   **평가 지표 (Metrics)**:
    *   **Average Reward**: 에피소드 당 평균 보상.
    *   **Episode Length**: 에피소드 지속 시간 (오래 버티는 것이 목표인 경우).
    *   **Prediction Error (FE)**: 관측 데이터에 대한 예측 오차 (Variational Free Energy).

## 3. 실험 과정 (Procedure)

1.  두 에이전트를 동일한 환경(Atari Breakout)에서 학습시킵니다.
2.  일정 에피소드마다 체크포인트를 저장하고 평가를 수행합니다.
3.  학습 곡선(Learning Curve)을 기록하여 수렴 속도와 최종 성능을 비교합니다.
4.  (선택 사항) 노이즈를 주입하거나 환경 파라미터를 변경하여 강건성을 테스트합니다.

## 4. 실행 방법 (Execution)

```bash
# 비교 실험 실행 (결과는 results/comparison_plots.png 등에 저장됨)
export PYTHONPATH=$PYTHONPATH:.
python src/experiments/compare_performance.py
```

## 5. 예상 결과 (Expected Results)

*   **초기 학습**: Flat 모델이 더 단순하므로 초기에는 빠르게 학습할 수 있습니다.
*   **장기 성능**: Hierarchical 모델이 복잡한 패턴을 파악하고 장기적인 전략을 세우므로, 후반부로 갈수록 더 높은 보상을 얻을 것으로 예상됩니다.
*   **계층적 이점**: 공이 빨라지거나 블록 패턴이 복잡해질수록 계층적 모델의 성능 우위가 뚜렷해질 것입니다.
