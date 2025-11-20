# 모델 아키텍처 (Model Architecture)

이 모델은 딥러닝의 계층적 구조와 Active Inference의 확률적 추론을 결합한 **계층적 POMDP (Hierarchical POMDP)** 구조를 가집니다.

## 1. 주요 구성 요소 (Components)

생성 모델(Generative Model)은 각 계층(Level)마다 다음과 같은 텐서(Tensor)들로 정의됩니다.

* **A (Likelihood)**: 은닉 상태(Hidden State)를 관측(Outcome) 또는 하위 레벨의 상태로 매핑합니다.
* **B (Transitions)**: 은닉 상태 간의 전이(Transition)를 정의합니다. 이는 *경로(Path)*에 의해 조건부로 결정됩니다.
* **C (Preferences)**: 결과에 대한 사전 선호도(Prior Preferences)를 나타냅니다. 목표 지향적 계획(Goal-directed Planning)에 사용됩니다.
* **D (Initial States)**: 초기 은닉 상태에 대한 사전 믿음(Prior)입니다.
* **E (Paths)**: 경로(전이들의 시퀀스)에 대한 사전 믿음입니다.

## 2. 딥러닝과 Active Inference의 결합

### 계층적 내포 및 재규격화 (Hierarchical Nesting & Renormalization)

* **Top-Down Generation**: 레벨 $L$의 하나의 상태는 레벨 $L-1$의 *궤적(Trajectory)*, 즉 상태들의 시퀀스를 생성합니다.
* **Spatial Blocking**: 이미지 처리와 같은 공간적 작업에서, 레벨 $L$의 상태는 레벨 $L-1$의 상태 그리드(예: $2 \times 2$)를 생성할 수 있습니다.
* 이 구조는 Deep CNN(Convolutional Neural Network)과 유사하지만, 결정론적 연산 대신 확률적 생성 과정을 거친다는 점이 다릅니다.

### 메시지 패싱을 통한 추론 (Inference as Message Passing)

* 역전파(Backpropagation) 대신 **변분 메시지 패싱(Variational Message Passing)**을 사용합니다.
* 믿음(Beliefs)은 상향식(Bottom-up) 인식과 하향식(Top-down) 예측을 통해 동시에 전파됩니다.

### 계획 모듈 (Planning Module)

* 별도의 계획 모듈이 존재하는 것이 아니라, **"추론으로서의 계획(Planning as Inference)"** 개념을 사용합니다.
* 모델은 데이터를 가장 잘 설명하면서도 선호하는 결과로 이끄는 최적의 "경로($u$)"를 추론합니다.

## 3. "Scale-free"의 의미

"Scale-free"는 모델의 구조와 역학(Dynamics)이 모든 계층에서 보편적이고 자기 유사적(Self-similar)임을 의미합니다.

1. **역학의 불변성 (Invariance of Dynamics)**: 믿음 업데이트(VFE 최소화)와 계획(EFE 최소화)을 위한 수학적 규칙은 **모든 계층에서 동일**합니다. 픽셀을 처리하는 코드와 계획을 수립하는 코드는 근본적으로 같습니다.
2. **시간적 재규격화 (Temporal Renormalization)**:
    * **Level 1**: 밀리초(ms) 단위의 빠른 역학.
    * **Level 2**: Level 1 상태들의 시퀀스를 나타내며, 더 느린 역학을 가집니다.
    * **Level 3**: Level 2 시퀀스들의 시퀀스(서사/장기 계획)를 나타냅니다.
    * 상위 레벨로 갈수록 시간이 "압축(Coarse-graining)"되어 더 긴 시간 범위를 다루게 됩니다.
3. **프랙탈 구조 (Fractal Structure)**: 모델은 세상을 "입자들의 입자(Particles of particles)"로 봅니다. 한 스케일의 상태는 하위 스케일의 궤적을 담는 컨테이너 역할을 하며, 이를 통해 깊이의 제한 없이 확장 가능합니다.
