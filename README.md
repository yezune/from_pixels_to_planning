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

* [ ] **Phase 5: 논문 실험 재현 (Paper Reproduction)**
  * [x] **계층적 모델(Hierarchical Model) 확장**: 현재의 단일 계층(Flat) 모델을 다층 구조(Level 1: Pixels -> Level 2: Paths)로 확장 구현.
    * *필요 작업*: `src/models/hierarchical_agent.py` 생성, 상위 레벨의 VAE/Transition 모델 정의, 레벨 간 통신(Top-down prediction, Bottom-up error) 로직 구현.
  * [x] **계층적 학습 루프(Hierarchical Training Loop)**: Level 1과 Level 2 모델을 동시에 학습시키는 트레이너 구현.
    * *필요 작업*: `src/hierarchical_trainer.py` 구현, Level 2 VAE (MlpVAE) 구현, 통합 테스트.
  * [x] **MiniGrid 환경 실험**: 논문에서 제시하는 GridWorld 기반의 픽셀 네비게이션 실험 재현.
    * *필요 작업*: `gym-minigrid` 연동, 픽셀 관측(Pixel Observation) 래퍼 설정, 희소 보상(Sparse Reward) 문제 해결을 위한 탐색 전략 튜닝.
  * [x] **전문가 궤적 학습 (Expert Trajectory Learning)**: 성공적인 경로를 압축하여 상위 레벨의 계획(Plan)으로 학습하는 기능 구현.
    * *필요 작업*: 오프라인 데이터셋 수집(또는 사전 학습된 전문가 에이전트 활용), 상위 레벨의 'Path' 잠재 변수 학습 로직 구현.
  * [x] **Atari (Breakout/Pong) 실험**: 고차원 픽셀 입력과 빠른 동적 변화를 다루는 Atari 게임 실험.
    * *완료 작업*: `ale-py` 및 `AutoROM`을 이용한 Atari 환경 구축, `AtariPixelEnv` 래퍼 구현 (64x64 리사이징), TDD 기반 실험 코드 작성 (`src/experiments/atari_experiment.py`).
  * [x] **성능 비교 및 분석**: Flat 모델 vs Hierarchical 모델의 성능(성공률, 학습 속도) 및 계획 수립 능력 비교.
    * *완료 작업*: `ComparisonRunner` 구현, TDD 기반 비교 실험 스크립트 작성 (`src/experiments/run_comparison.py`), JSON 결과 저장 기능 구현.
  * [ ] **MNIST 분류 실험 (MNIST Classification)**: 정적 이미지에 대한 공간적 계층 구조(Spatial Renormalization) 학습 및 분류 실험 재현.
    * *필요 작업*: `src/experiments/mnist_experiment.py` 구현, Spatial RGM 모델 정의, 분류 정확도 측정.

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
