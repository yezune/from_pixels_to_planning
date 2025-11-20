# 논문 상세 분석: 주요 수식 및 그림 (Key Equations & Figures)

이 문서는 논문 **"From pixels to planning: Scale-free active inference"**의 핵심적인 수학적 공식과 주요 그림(Figure)들에 대한 설명을 담고 있습니다.

## 1. 주요 수식 (Key Equations)

논문의 핵심은 **자유 에너지 원리(Free Energy Principle)**를 계층적 모델에 적용하는 것입니다.

### 1.1. 생성 모델 (Generative Model)
에이전트는 세상에 대한 결합 확률 분포(Joint Probability Distribution)를 가집니다.

$$
P(\tilde{o}, \tilde{s}, \tilde{u}) = P(\tilde{s} | \tilde{u}) P(\tilde{o} | \tilde{s}) P(\tilde{u})
$$

*   $\tilde{o}$: 관측(Observations)의 시퀀스
*   $\tilde{s}$: 은닉 상태(Hidden States)의 시퀀스
*   $\tilde{u}$: 제어/정책(Control/Policy)의 시퀀스

### 1.2. 변분 자유 에너지 (Variational Free Energy, $F$)
지각(Perception)과 상태 추론(State Estimation)은 $F$를 최소화하는 과정입니다.

$$
F = \underbrace{\mathbb{E}_Q[\ln Q(s) - \ln P(s)]}_{\text{Complexity (KL Divergence)}} - \underbrace{\mathbb{E}_Q[\ln P(o|s)]}_{\text{Accuracy (Log-Likelihood)}}
$$

*   **의미**: 모델은 관측 데이터를 정확하게 설명(Accuracy)하면서도, 복잡하지 않은 설명(Complexity)을 선호합니다.

### 1.3. 기대 자유 에너지 (Expected Free Energy, $G$)
행동 선택(Action Selection)과 계획(Planning)은 미래의 예상되는 $F$, 즉 $G$를 최소화하는 과정입니다.

$$
G(\pi) = \underbrace{- \mathbb{E}_{\tilde{Q}}[\ln P(o|\pi)]}_{\text{Pragmatic Value (Extrinsic)}} + \underbrace{\mathbb{E}_{\tilde{Q}}[\ln Q(s|\pi) - \ln Q(s|o, \pi)]}_{\text{Epistemic Value (Intrinsic)}}
$$

*   **Pragmatic Value**: 에이전트가 선호하는 결과(Goal)를 얻을 확률을 높입니다.
*   **Epistemic Value**: 정보 이득(Information Gain)을 통해 불확실성을 줄입니다 (탐색).

---

## 2. 주요 그림 설명 (Key Figures Description)

논문에 등장하는 주요 도표들의 개념적 설명입니다.

### Figure 1: Renormalizing Generative Model (RGM)의 구조
*   **설명**: 이 그림은 단일 계층의 POMDP(Partially Observed Markov Decision Process)가 어떻게 상위 계층으로 확장되는지 보여줍니다.
*   **핵심 요소**:
    *   **하위 레벨 (Level 1)**: 픽셀 단위의 빠른 변화를 처리합니다.
    *   **상위 레벨 (Level 2)**: 하위 레벨의 상태들의 *시퀀스(Sequence)*를 하나의 *상태(State)*로 취급합니다.
    *   **Renormalization**: 시간과 공간을 압축(Coarse-graining)하여 상위 레벨로 전달하는 과정을 시각화합니다.

### Figure 2: 팩터 그래프 (Factor Graph)
*   **설명**: 변수들 간의 인과 관계를 나타내는 그래프입니다.
*   **구조**:
    *   $s_t$ (현재 상태)는 $s_{t-1}$ (이전 상태)와 $u_{t-1}$ (행동)에 의존합니다.
    *   $o_t$ (관측)는 $s_t$ (현재 상태)에서 생성됩니다.
    *   이 그래프는 계층적으로 쌓여 있어, 상위 레벨의 상태가 하위 레벨의 초기 조건이나 전이 확률에 영향을 미칩니다.

### Figure 3: MNIST 실험 (Spatial Renormalization)
*   **설명**: 정적인 이미지(MNIST 숫자)를 공간적으로 분해하여 계층적으로 처리하는 과정을 보여줍니다.
*   **과정**:
    1.  이미지를 작은 패치(Patch)로 나눕니다.
    2.  하위 레벨은 각 패치의 특징을 학습합니다.
    3.  상위 레벨은 패치들의 배치를 보고 전체 숫자(Digit)를 추론합니다.
*   **결과**: 모델이 숫자를 분류할 뿐만 아니라, 상위 레벨의 개념(숫자 '5')으로부터 하위 레벨의 픽셀 이미지를 생성(Generation)해내는 것을 보여줍니다.

### Figure 4: Atari Breakout 실험 (Temporal Renormalization)
*   **설명**: 동적인 환경에서 시간적 계층 구조가 어떻게 작동하는지 보여줍니다.
*   **비교**:
    *   **Flat Agent**: 공의 움직임에 즉각적으로 반응하지만, 장기적인 계획을 세우지 못해 공을 놓치는 경우가 많습니다.
    *   **Hierarchical Agent**: "공을 받아낸다"는 상위 목표를 설정하고, 이를 달성하기 위한 하위 행동 시퀀스를 생성하여 더 안정적으로 게임을 수행합니다.

### Figure 5: 성능 비교 (Performance Comparison)
*   **설명**: 학습 에피소드가 진행됨에 따라 Flat 모델과 Hierarchical 모델의 누적 보상(Cumulative Reward)을 비교한 그래프입니다.
*   **결과**: Hierarchical 모델이 초기에는 학습이 느릴 수 있으나(복잡성 때문), 장기적으로는 더 높은 점수에 수렴하고 노이즈에 강건함을 보여줍니다.
