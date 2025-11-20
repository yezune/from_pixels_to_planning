# Experiment 2: Atari Breakout (Hierarchical Active Inference)

## 1. 이론적 배경 (Theoretical Background)

이 실험은 논문의 핵심 주제인 **Scale-Free Active Inference**를 검증합니다. 복잡하고 고차원적인 환경(Atari)에서는 단일 계층 모델만으로는 효율적인 계획(Planning)이 어렵습니다. 따라서 시간적/공간적 추상화 수준이 다른 계층적 모델을 도입합니다.

### 계층적 구조 (Hierarchical Structure)
*   **Level 1 (Bottom)**: 픽셀 단위의 빠른 변화를 처리합니다.
    *   입력: 64x64 RGB 이미지.
    *   출력: 원시 행동 (Joystick movement).
    *   역할: 상위 레벨의 지령(Goal)을 달성하기 위한 구체적인 행동 생성.
*   **Level 2 (Top)**: 추상적인 상태와 긴 시간 간격의 계획을 처리합니다.
    *   입력: Level 1의 잠재 상태($z^1$).
    *   출력: Level 1을 위한 목표 상태($z^1_{goal}$) 또는 추상적 행동.
    *   역할: 전체적인 에피소드 흐름을 파악하고 장기적인 목표 설정.

## 2. 실험 설정 (Setup)

*   **환경 (Environment)**: `AtariPixelEnv` (BreakoutNoFrameskip-v4)
    *   OpenAI Gym의 Atari 환경을 래핑.
    *   전처리: 210x160 이미지를 64x64로 리사이징 및 정규화.
*   **모델 (Model)**: `HierarchicalAgent`
    *   **Level 1**: CNN-VAE + Transition Model.
    *   **Level 2**: MLP-VAE (Level 1의 잠재 벡터를 입력으로 받음) + Transition Model.

## 3. 실험 과정 (Procedure)

1.  **계층적 데이터 수집**: 환경 상호작용을 통해 픽셀 데이터와 행동을 수집합니다.
2.  **Level 1 학습**: 픽셀 재구성 및 1-step 예측을 학습합니다.
3.  **Level 2 학습**: Level 1에서 인코딩된 잠재 벡터($z^1$)들의 시퀀스를 학습합니다. Level 2는 $z^1$의 궤적을 예측하거나 압축합니다.
4.  **Top-Down Planning**: (심화 단계) Level 2가 목표 상태를 내려보내면, Level 1은 그 상태와의 차이(Prediction Error)를 최소화하는 행동을 선택합니다.

## 4. 실행 방법 (Execution)

```bash
# Atari 실험 실행
export PYTHONPATH=$PYTHONPATH:.
python src/experiments/atari_experiment.py
```

## 5. 예상 결과 (Expected Results)

*   **Abstraction**: Level 2의 잠재 공간은 공의 구체적인 위치보다는 "공이 위로 올라가는 중", "블록이 깨짐"과 같은 더 추상적인 정보를 담게 됩니다.
*   **Robustness**: 계층적 모델은 노이즈나 작은 변화에 더 강건하며, 장기적인 보상을 얻는 전략을 수립할 수 있습니다.
