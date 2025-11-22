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

## 4. 구현된 아키텍처 (Implemented Architecture)

본 프로젝트에서 구현된 실제 아키텍처는 다음과 같습니다.

### 4.1 3-Level Hierarchical RGM

논문의 개념을 구체화하여 3단계 계층 구조를 구현했습니다.

| Level | 역할 | Latent Dim | Temporal Resolution (τ) | 구성 요소 |
|-------|------|------------|-------------------------|-----------|
| **Level 2** | **Path / Goal** | 8D | 16 steps | VAE (16D→8D) + Transition |
| **Level 1** | **Feature / Sub-goal** | 16D | 4 steps | VAE (32D→16D) + Transition |
| **Level 0** | **Pixel / Action** | 32D | 1 step | VAE (Pixel→32D) + Transition |

* **Level 0 (Pixel Level)**:
    * 매 스텝(τ=1)마다 픽셀 입력을 처리하고 즉각적인 액션을 선택합니다.
    * VAE는 64x64 RGB 이미지를 32차원 잠재 공간으로 압축합니다.
* **Level 1 (Feature Level)**:
    * 4 스텝(τ=4)마다 업데이트되며, Level 0의 상태를 요약하고 단기 목표(Sub-goal)를 설정합니다.
    * Level 0의 32D 상태를 16D로 압축합니다.
* **Level 2 (Path Level)**:
    * 16 스텝(τ=16)마다 업데이트되며, 장기적인 목표(Long-term Goal)를 설정합니다.
    * Level 1의 16D 상태를 8D로 압축합니다.

### 4.2 Planning & Action Selection

계층적 구조를 활용한 Planning은 다음과 같이 이루어집니다.

1. **Top-Down Goal Setting**:
    * 상위 레벨(Level 2)에서 장기 목표를 설정하면, 이는 하위 레벨(Level 1)의 사전 믿음(Prior)으로 작용합니다.
    * Level 1은 이 목표를 달성하기 위한 세부 목표를 생성하여 Level 0에 전달합니다.
2. **MCTS (Monte Carlo Tree Search) Integration**:
    * **Level 1 Latent Space**에서 MCTS를 수행하여 효율적인 계획을 수립합니다.
    * 픽셀 공간이 아닌 압축된 잠재 공간에서 시뮬레이션하므로 계산 효율성이 높습니다.
    * 4 스텝마다 Re-planning을 수행하여 변화하는 환경에 적응합니다.
3. **Action Selection**:
    * Level 0는 상위 레벨의 가이드와 현재 관측을 바탕으로 최종 액션을 선택합니다.
    * Random, Flat, Hierarchical, MCTS 등 다양한 정책을 비교 실험할 수 있습니다.

### 4.3 Multi-Game Validation

이 아키텍처는 특정 게임에 종속되지 않는 일반화 능력을 가집니다.

* **Atari Breakout**: 벽돌 깨기, 공의 궤적 예측 및 전략적 패들 이동. (Hierarchical Planning 효과 큼)
* **Atari Pong**: 상대 AI와의 대결, 빠른 반응 속도 중요. (Random/Reactive Policy가 효과적일 수 있음)
* **Bouncing Ball**: 물리 법칙 학습 및 예측 검증용 Toy Environment.

### 4.4 Hybrid Architecture for Reactive Games (New)

Pong과 같이 빠른 반응속도(High Reactivity)와 속도 정보(Velocity)가 중요한 게임에서의 성능 한계를 극복하기 위해, **Hybrid Architecture**를 도입합니다.

* **Reactive Component (Fast Path)**:
  * **Frame Stacking**: 4개의 연속된 프레임을 입력으로 사용하여 공의 속도와 방향 정보를 포착합니다.
  * **Model-Free Policy (DQN)**: 복잡한 생성 모델을 거치지 않고, 입력에서 바로 액션을 매핑하는 가벼운 신경망을 사용합니다.
  * **역할**: 공이 패들에 가까워지는 등 즉각적인 반응이 필요한 상황(Emergency)을 처리합니다.
* **Integration Strategy**:
  * 평상시에는 **Hierarchical Planner (Slow Path)**가 장기적인 전략을 수립합니다.
  * 긴급 상황이나 빠른 반응이 요구되는 구간에서는 **Reactive Component**가 제어권을 가집니다.
  * 이는 인간의 인지 과정(System 1: 직관/반응 vs System 2: 추론/계획)과 유사한 구조입니다.

## 5. 데이터 흐름 (Data Flow)

1. **Bottom-Up Inference**:
   `Observation (Pixel) → Level 0 Latent → Level 1 Latent → Level 2 Latent`
   (현재 상태를 모든 레벨에서 추론)

2. **Top-Down Prediction**:
   `Level 2 Prior → Level 1 Prior → Level 0 Prior`
   (상위 레벨의 예측이 하위 레벨의 가이드가 됨)

3. **Action Execution**:
   `Level 0 Latent + Goal → Action`
   (최하위 레벨에서 환경과 상호작용)

