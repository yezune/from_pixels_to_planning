# From Pixels to Planning: Scale-Free Active Inference

## 📖 프로젝트 개요 (Project Overview)

이 저장소는 논문 **["From pixels to planning: Scale-free active inference"](https://arxiv.org/abs/2407.20292)**를 체계적으로 분석하고, 이를 바탕으로 실제 코드로 구현하여 이해를 돕기 위해 만들어졌습니다.

이 프로젝트의 주된 목적은 복잡한 Active Inference(능동적 추론) 이론이 픽셀 단위의 시각적 입력에서부터 상위 수준의 계획(Planning)까지 어떻게 연결되는지 설명하고, 이를 입증하는 실험 코드를 제공하는 것입니다.

## 🎯 프로젝트 목표 (Goals)

1. **논문 심층 분석**: 논문의 핵심 아이디어, 수식, 모델 구조를 누구나 이해할 수 있도록 문서화합니다.
2. **코드 구현 (Implementation)**: 논문에서 제안하는 Scale-free Active Inference 모델을 Python으로 구현합니다.
3. **실험 및 검증**: 구현된 모델을 통해 논문의 결과를 재현하거나, 간단한 환경에서 동작 원리를 시각화합니다.

## 📂 폴더 구조 (Repository Structure)

```bash
.
├── README.md           # 프로젝트 메인 문서
├── docs/               # 논문 분석 자료 및 설명 문서
│   ├── summary.md      # 논문 요약
│   ├── math.md         # 수식 유도 및 설명
│   └── architecture.md # 모델 아키텍처 설명
├── src/                # 소스 코드
│   ├── models/         # Active Inference 모델 구현
│   ├── envs/           # 실험 환경 (GridWorld, Doom 등)
│   └── utils/          # 유틸리티 함수
├── notebooks/          # Jupyter Notebook 실험 및 시각화
└── references/         # 참고 문헌 및 관련 자료
```

## 💡 핵심 개념 (Key Concepts)

이 프로젝트에서 다루는 주요 개념들은 다음과 같습니다:

* **Active Inference (능동적 추론)**: 지각(Perception)과 행동(Action)을 자유 에너지(Free Energy) 최소화 과정으로 통합하는 프레임워크.
* **Scale-Free Dynamics**: 시간적/공간적 스케일에 구애받지 않는 계층적 처리 구조.
* **Deep Active Inference**: 딥러닝 신경망을 사용하여 고차원 입력(Pixels)을 처리하고 계획(Planning)을 수행하는 방법.
* **Generative Models**: 에이전트가 세상을 이해하고 예측하기 위해 내부적으로 구축하는 모델.
* **Planning in Latent Space (잠재 공간에서의 계획)**: 학습된 저차원 잠재 공간에서 효율적으로 다단계 계획을 수립하는 방법.
  * **MCTS (Monte Carlo Tree Search)**: 불확실성 하에서 최적 행동 시퀀스를 탐색하는 트리 탐색 알고리즘
  * **Trajectory Optimization**: 미분 가능한 전이 모델을 통해 경로를 최적화하는 경사 기반 방법

## 🚀 로드맵 (Roadmap)

* [x] **Phase 1: 이론 학습 및 정리**
  * [x] 논문 초록 및 서론 분석
  * [x] 핵심 수식 (Free Energy, Expected Free Energy) 정리
  * [x] 모델 아키텍처 다이어그램 작성

* [x] **Phase 2: 기본 환경 구축**
  * [x] 실험을 위한 시뮬레이션 환경 설정 (예: MiniGrid, VizDoom)
  * [x] 데이터 전처리 파이프라인 구축

* [x] **Phase 3: 모델 구현**
  * [x] VAE (Variational Autoencoder) 기반의 지각 모델 구현
  * [x] RNN/LSTM 기반의 전이(Transition) 모델 구현
  * [x] Action Selection 및 Planning 알고리즘 구현
  * [x] **Planning Module (계획 모듈)**: 학습된 잠재 공간에서의 계획 수립
    * **MCTS (Monte Carlo Tree Search)**: UCB1 기반 트리 탐색으로 다단계 계획 수립
    * **Trajectory Optimization**: Gradient 기반 및 Cross-Entropy Method를 이용한 경로 최적화

* [x] **Phase 4: 실험 및 시각화**
  * [x] 학습 과정 시각화 (Loss, Free Energy)
  * [x] 에이전트의 행동 및 계획 과정 시각화
  * [x] 실제 학습 루프 구현 및 BouncingBall 환경 테스트

* [x] **Phase 5: 논문 실험 재현 (Paper Reproduction)** ✅ **완료!**
  * [x] **계층적 모델(Hierarchical Model) 확장**: 3-Level 구조 (Level 0: Pixels → Level 1: Features → Level 2: Paths) 완전 구현
  * [x] **계층적 학습 루프(Hierarchical Training Loop)**: 전체 3-Level hierarchy 학습 파이프라인 구현 및 학습 완료
    * *완료*: Level 1 VAE (Loss 29.24), Level 1 Transition (Loss 1.008)
    * *완료*: Level 2 VAE (Loss 14.35), Level 2 Transition (Loss 1.623)
    * *학습 시간*: 2.5분 (매우 효율적!)
  * [x] **시간적 추상화(Temporal Abstraction) 검증**: 상위 레벨이 더 긴 미래를 더 정확하게 예측
    * *완료*: Level 1 (τ=4, MSE 0.980) vs Level 2 (τ=16, MSE 0.922) - Level 2가 더 우수!
  * [x] **계층적 Planning 성능 실증**: 3가지 방법 비교 (Random vs Flat vs Hierarchical)
    * *완료*: Hierarchical이 Random 대비 **45.5% 성능 향상**
    * *완료*: Flat planning은 18.2% 성능 저하 (단일 레벨의 한계)
    * *완료*: 최대 보상 4.0 달성 (다른 방법들은 최대 3.0)
  * [x] **Atari (Breakout) 실험**: 고차원 픽셀 입력과 빠른 동적 변화를 다루는 Atari 게임 실험
    * *완료*: VAE 학습 (PSNR 34.41 dB, 99.52% accuracy)
    * *완료*: Transition 학습 (MSE 0.000710)
    * *완료*: 계층적 모델 학습 및 Planning 테스트
  * [x] **성능 비교 및 분석**: Scale-free dynamics의 실제 효과 검증
    * *완료*: 1,536x 압축 (12,288D → 8D) 달성
    * *완료*: 계층적 Planning이 실제로 더 나은 성능 달성
    * *완료*: 시각화 및 상세 결과 문서화

## 🛠 설치 (Installation)

```bash
# 가상 환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 🧪 실험 가이드 (Experiments Guide)

이 프로젝트는 단계별로 다양한 실험을 제공합니다. 모든 명령어는 프로젝트 루트 디렉토리에서 실행해야 합니다.

### 1. 기초 실험: Bouncing Ball (Phase 4)
가장 기본적인 Active Inference 에이전트가 간단한 물리 환경(튀기는 공)을 학습하는지 확인합니다.
```bash
export PYTHONPATH=$PYTHONPATH:.
python src/main.py
```

### 2. 심화 실험: Atari Breakout (Phase 5)
고차원 픽셀 입력(Atari)에서 계층적(Hierarchical) 모델을 학습시킵니다.
```bash
export PYTHONPATH=$PYTHONPATH:.
python src/experiments/atari_experiment.py
```
*   **참고**: 스크립트 내 `num_epochs` 변수를 수정하여 학습 길이를 조절할 수 있습니다.

### 3. 비교 실험: Flat vs Hierarchical
단일 계층(Flat) 모델과 계층적(Hierarchical) 모델의 성능을 비교합니다.
```bash
export PYTHONPATH=$PYTHONPATH:.
python src/experiments/run_comparison.py --episodes 10
```
*   `--episodes`: 평가할 에피소드 수 (기본값: 5)
*   `--flat_ckpt`: Flat 모델 체크포인트 경로 (선택 사항)
*   `--hier_ckpt`: Hierarchical 모델 체크포인트 경로 (선택 사항)

## ✅ 테스트 실행 (Running Tests)

실험 코드가 정상적으로 작동하는지 검증하기 위해 작성된 테스트 코드를 실행할 수 있습니다.

### Atari 실험 검증
Atari 환경 설정, 모델 초기화, 학습 루프가 정상 작동하는지 확인합니다.
```bash
python -m unittest tests/test_phase5_atari_experiment.py
```

### 비교 실험 검증
비교 실험 러너(ComparisonRunner)와 평가 로직이 정상 작동하는지 확인합니다.
```bash
python -m unittest tests/test_phase5_comparison.py
```

### 전체 인수 테스트 (Acceptance Test)
전체 파이프라인(Phase 4, Phase 5, 비교 실험)이 정상적으로 작동하는지 한 번에 검증합니다.
```bash
python -m unittest tests/test_acceptance.py
```

## 🧪 실험 목록 (Experiments)

이 프로젝트는 다음의 실험들을 통해 이론을 검증합니다. 각 실험에 대한 자세한 내용은 `docs/experiments/` 폴더의 문서를 참고하세요.

1. **[RGM Fundamentals](docs/experiments/01_rgm_fundamentals.md)**: Renormalization Group Method의 핵심 개념 시각화 (추상화, 생성, 지역성).
2. **[MNIST Classification](docs/experiments/02_mnist_classification.md)**: 정적 이미지에 대한 공간적 계층 구조 학습 및 분류.
3. **[Bouncing Ball (Basic)](docs/experiments/03_bouncing_ball.md)**: 단일 계층 Active Inference 모델의 기초 검증.
4. **[Atari Breakout (Planning)](docs/experiments/04_atari_breakout.md)**: **Planning 모듈 통합** - MCTS와 Trajectory Optimization을 이용한 잠재 공간 계획.
5. **[Performance Comparison](docs/experiments/05_performance_comparison.md)**: Flat vs Hierarchical 모델의 성능 비교.

## 📊 시각화 (Visualization)

Jupyter Notebook을 통해 실험 결과를 시각적으로 확인할 수 있습니다.

*   **`notebooks/experiments_visualization.ipynb`**: Bouncing Ball 및 Atari 실험의 학습 과정과 모델 재구성 결과, 성능 비교 그래프를 제공합니다.

## 📚 참고 문헌 (References)

*   [From pixels to planning: Scale-free active inference](https://arxiv.org/abs/2407.20292) (ArXiv)
*   [Active Inference Institute](https://www.activeinference.org/)
