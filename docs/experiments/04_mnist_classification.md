# Experiment 4: MNIST Classification (Hierarchical Static Inference)

## 1. 이론적 배경 (Theoretical Background)

이 실험은 논문에서 소개된 **Renormalizing Generative Model (RGM)**의 가장 기초적인 적용 사례로, 정적인 이미지(Static Image)인 MNIST 숫자 데이터를 계층적으로 처리하여 분류(Classification)하는 과제를 수행합니다.

### 공간적 재규격화 (Spatial Renormalization)
RGM은 시간적 시퀀스뿐만 아니라 공간적 패턴에도 적용될 수 있습니다.
*   **Level 1 (Pixels)**: 이미지를 작은 패치(예: 2x2 또는 4x4)로 분할하여 처리합니다.
*   **Level 2 (Features)**: 하위 레벨의 패치들이 모여 형성하는 지역적 특징(Stroke, Curve)을 인코딩합니다.
*   **Level 3 (Digit Identity)**: 전체적인 숫자의 형상과 클래스(0~9)를 나타내는 최상위 개념을 추론합니다.

이 과정은 픽셀에서 시작하여 점차 상위 개념으로 나아가는 **Bottom-Up Inference**와, 상위 개념이 하위 픽셀을 예측하는 **Top-Down Generation**의 상호작용으로 이루어집니다.

## 2. 실험 설정 (Setup)

*   **데이터셋 (Dataset)**: MNIST (28x28 Grayscale Images of Handwritten Digits).
*   **모델 (Model)**: Spatial RGM (Hierarchical Discrete State-Space Model).
    *   **Input**: 28x28 이미지를 Flatten하거나 Patch로 분할하여 입력.
    *   **Latent States**: 각 계층별 이산 잠재 변수(Discrete Latent Variables).
    *   **Output**: 숫자 클래스 (0-9) 및 이미지 재구성(Reconstruction).

## 3. 실험 과정 (Procedure)

1.  **계층적 학습 (Hierarchical Learning)**:
    *   하위 레벨부터 순차적으로 학습하거나, 전체 모델을 End-to-End로 학습합니다.
    *   각 레벨은 자신의 상위 레벨로부터의 예측(Prior)과 하위 레벨로부터의 입력(Likelihood)을 결합하여 사후 확률(Posterior)을 계산합니다.
2.  **분류 (Classification)**:
    *   테스트 이미지가 주어졌을 때, 최상위 레벨의 잠재 상태(Latent State)가 어떤 숫자 클래스에 해당하는지 확인합니다.
    *   Active Inference 관점에서는 "어떤 숫자인지 맞추는 것"을 하나의 행동(Action)이나 내부 상태 추론으로 볼 수 있습니다.
3.  **생성 (Generation)**:
    *   특정 숫자 클래스(예: "5")를 최상위 레벨에 고정하고 Top-Down으로 신호를 내려보내 이미지를 생성합니다.

## 4. 실행 방법 (Execution)

```bash
# MNIST 분류 실험 실행
export PYTHONPATH=$PYTHONPATH:.
python src/experiments/mnist_experiment.py
```

## 5. 예상 결과 (Expected Results)

*   **Classification Accuracy**: 기존 CNN과 유사하거나 경쟁력 있는 분류 정확도를 달성해야 합니다.
*   **Generative Capability**: 단순히 분류만 하는 것이 아니라, 특정 숫자를 그리는(생성하는) 능력을 보여줍니다.
*   **Robustness**: 노이즈가 섞인 이미지나 일부가 가려진 이미지에 대해서도, 상위 레벨의 사전 지식(Prior)을 통해 강건하게 추론할 수 있습니다.
