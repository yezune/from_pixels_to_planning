# 핵심 수식 (Key Mathematical Formulations)

이 논문은 **자유 에너지 원리(Free Energy Principle)**를 기반으로 하며, 추론(Inference)과 의사결정(Decision-making)을 위해 두 가지 주요 수식을 사용합니다.

## 1. 변분 자유 에너지 (Variational Free Energy, VFE) - $F$

VFE는 **지각(Perception)** 및 **추론(Inference)**을 위한 목적 함수입니다. VFE를 최소화하는 것은 관측 데이터에 대한 모델의 증거(Model Evidence)를 최대화하는 것과 같습니다.

$$
F = \underbrace{D_{KL}[Q(s, u, a) \parallel P(s, u, a)]}_{\text{Complexity}} - \underbrace{\mathbb{E}_Q[\ln P(o \mid s, u, a)]}_{\text{Accuracy}}
$$

* $Q(s, u, a)$: 숨겨진 상태($s$), 경로($u$), 파라미터($a$)에 대한 근사 사후 믿음(Approximate Posterior Beliefs).
* $P(s, u, a)$: 사전 믿음(Prior Beliefs).
* $P(o \mid \dots)$: 관측($o$)에 대한 우도(Likelihood).
* **해석**: 에이전트는 데이터를 정확하게 설명하면서도(Accuracy 최대화), 기존의 믿음에서 너무 벗어나지 않는 단순한 설명(Complexity 최소화)을 찾으려 합니다.

## 2. 기대 자유 에너지 (Expected Free Energy, EFE) - $G$

EFE는 **계획(Planning)** 및 **능동적 학습(Active Learning)**을 위한 목적 함수입니다. 미래에 예상되는 VFE를 평가합니다.

$$
G(u) = \underbrace{- \mathbb{E}_{Q_u}[\ln Q(a \mid s_{\tau+1}, o_{\tau+1}, u) - \ln Q(a \mid s_{\tau+1}, u)]}_{\text{Information Gain (Epistemic)}} \underbrace{- \mathbb{E}_{Q_u}[\ln P(o_{\tau+1} \mid c)]}_{\text{Expected Value (Pragmatic)}}
$$

* **계획(Planning)의 관점**: 에이전트는 $G(u)$를 최소화하는 경로($u$)를 선택합니다. 이는 불확실성을 줄이기 위한 탐색(Information Gain)과 선호하는 결과($c$)를 얻기 위한 활용(Expected Value) 사이의 균형을 맞춥니다.
* **능동적 학습(Active Learning)의 관점**: 모델은 파라미터 업데이트가 EFE를 줄이는지(즉, 유의미한 정보 이득을 제공하는지)에 따라 학습 여부를 결정합니다.

## 3. 수식의 의미와 적용

이 두 수식은 모델의 모든 계층에서 동일하게 적용됩니다.

* **하위 레벨**: 픽셀 데이터를 처리하며 현재 상태를 추론할 때 VFE를 최소화합니다.
* **상위 레벨**: 장기적인 계획을 세울 때 EFE를 최소화하는 경로를 선택합니다.
* **Scale-free**: 수식의 형태가 계층에 상관없이 일정하므로, 하나의 알고리즘으로 전체 시스템을 구동할 수 있습니다.
