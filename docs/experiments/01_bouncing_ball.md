# Experiment 1: Bouncing Ball (Basic Active Inference)

## 1. 이론적 배경 (Theoretical Background)

이 실험은 **Active Inference**의 가장 기초적인 형태를 검증하기 위한 "Worked Example"입니다. Active Inference 에이전트는 세상을 인식(Perception)하고 행동(Action)하는 과정을 **자유 에너지(Free Energy) 최소화** 문제로 풉니다.

### 핵심 구성 요소
1.  **Generative Model (생성 모델)**: 에이전트가 세상이 어떻게 작동하는지 믿는 내부 모델입니다.
    *   **VAE (Variational Autoencoder)**: 픽셀 이미지($o$)로부터 잠재 상태($z$)를 추론합니다. (Perception)
    *   **Transition Model**: 현재 상태($z_t$)와 행동($a_t$)으로부터 다음 상태($z_{t+1}$)를 예측합니다. (Dynamics)
2.  **Variational Free Energy (VFE)**: 지각의 놀라움(Surprise)에 대한 상한선(Upper Bound)입니다. 이를 최소화함으로써 에이전트는 관측 데이터를 가장 잘 설명하는 잠재 상태를 찾습니다.

## 2. 실험 설정 (Setup)

*   **환경 (Environment)**: `BouncingBallEnv`
    *   32x32 픽셀 크기의 2D 공간에서 공이 벽에 튀기는 물리 시뮬레이션.
    *   관측(Observation): (3, 32, 32) RGB 이미지.
    *   행동(Action): 상, 하, 좌, 우, 정지 (Discrete 5).
*   **모델 (Model)**: `ActiveInferenceAgent` (Flat)
    *   Encoder/Decoder: CNN 기반 VAE.
    *   Transition: MLP/GRU 기반 상태 전이 모델.

## 3. 실험 과정 (Procedure)

1.  **데이터 수집**: 에이전트가 환경과 상호작용하며 `(obs, action, reward, next_obs)` 데이터를 Replay Buffer에 저장합니다.
2.  **VAE 학습**: 관측 이미지를 잠재 공간으로 압축하고 다시 복원하는 능력을 학습합니다. (Reconstruction Loss 최소화)
3.  **Transition 학습**: 잠재 공간 상에서 내 행동에 따라 상태가 어떻게 변하는지 학습합니다. (Prediction Error 최소화)
4.  **행동 선택**: (현재 구현에서는 단순화됨) 미래의 기대 자유 에너지(EFE)를 최소화하는 행동을 선택합니다.

## 4. 실행 방법 (Execution)

```bash
# 기본 실행 (메인 스크립트)
export PYTHONPATH=$PYTHONPATH:.
python src/main.py

# 테스트 코드로 파이프라인 검증
python -m unittest tests/test_acceptance.py
```

## 5. 예상 결과 (Expected Results)

*   **Reconstruction**: 학습이 진행됨에 따라 VAE가 노이즈 섞인 공의 이미지를 선명하게 복원해야 합니다.
*   **Loss**: VAE Loss와 Transition Loss가 에포크가 지날수록 감소해야 합니다.
*   **Dynamics**: 에이전트는 공이 벽에 부딪혔을 때 튀어 나가는 물리를 잠재 공간 내에서 예측할 수 있게 됩니다.
