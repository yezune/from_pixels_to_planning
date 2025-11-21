# Experiment 4: Atari Breakout (Planning in Latent Space)

## 1. 이론적 배경 (Theoretical Background)

이 실험은 **Planning in Latent Space (잠재 공간에서의 계획)**을 검증합니다. 고차원 픽셀 공간에서 직접 계획을 수립하는 것은 계산 비용이 매우 높지만, 학습된 저차원 잠재 공간에서는 효율적인 다단계 계획이 가능합니다.

### Planning 모듈 (Planning Module)

본 실험에서는 두 가지 계획 알고리즘을 구현하여 비교합니다:

#### 1. MCTS (Monte Carlo Tree Search)
*   **원리**: UCB1(Upper Confidence Bound) 기반 트리 탐색으로 불확실성과 보상을 균형있게 고려
*   **장점**: 
    *   이산 행동 공간에서 강력
    *   확률적 환경에 강건
    *   다양한 경로 탐색 (Exploration)
*   **단점**: 
    *   많은 시뮬레이션 필요
    *   미분 불가능

#### 2. Trajectory Optimization (경로 최적화)
*   **원리**: 미분 가능한 전이 모델을 통해 행동 시퀀스를 경사 하강법으로 최적화
*   **구현 방법**:
    *   **Gradient-based**: Gumbel-Softmax를 이용한 연속 이완(Continuous Relaxation)으로 경사 계산
    *   **Cross-Entropy Method (CEM)**: 샘플링 기반 최적화로 엘리트 행동 시퀀스 선택
*   **장점**: 
    *   빠른 최적화 속도
    *   미분 가능한 목적 함수 활용
*   **단점**: 
    *   국소 최소값(Local Minima)에 빠질 수 있음
    *   이산 행동의 경우 이완(Relaxation) 필요

## 2. 실험 설정 (Setup)

* **환경 (Environment)**: `AtariPixelEnv` (BreakoutNoFrameskip-v4)
  * OpenAI Gym의 Atari 환경을 래핑
  * 전처리: 210x160 이미지를 64x64 RGB로 리사이징 및 정규화
* **모델 (Model)**: `ActiveInferenceAgent` with Planning
  * **VAE**: CNN 기반 변분 오토인코더 (3채널 RGB → 32차원 잠재 공간)
  * **Transition Model**: GRU 기반 전이 모델 (다음 상태 예측)
  * **Planning Module**: MCTS 또는 Trajectory Optimizer 선택 가능

## 3. 실험 과정 (Procedure)

1. **모델 초기화**: VAE와 Transition Model을 Atari 이미지 크기에 맞게 설정
2. **Reactive Agent 실행**: 1-step lookahead로 Expected Free Energy 최소화 (Baseline)
3. **MCTS Planning**: 10 시뮬레이션, 깊이 5로 트리 탐색하여 행동 선택
4. **Trajectory Optimization**: 경사 하강법으로 5-step 행동 시퀀스 최적화
5. **성능 비교**: 각 방법의 누적 보상, 실행 시간, 안정성 비교

## 4. 실행 방법 (Execution)

주피터 노트북을 통해 실험을 수행합니다:

```bash
jupyter notebook notebooks/04_atari_breakout.ipynb
```

또는 명령줄에서 실행:

```bash
export PYTHONPATH=$PYTHONPATH:.
python src/experiments/atari_experiment.py
```

## 5. 예상 결과 (Expected Results)

* **Planning 효과**: MCTS와 Trajectory Optimization은 Reactive 에이전트보다 높은 보상 달성
* **계산 효율성**: 잠재 공간(32차원)에서의 계획은 픽셀 공간(64×64×3)보다 훨씬 빠름
* **안정성**: Planning 에이전트는 더 일관된 행동 패턴 보임
* **Trade-off**: MCTS는 느리지만 다양한 경로 탐색, Trajectory Opt는 빠르지만 국소 최소값 가능

**Note**: 이 실험은 학습되지 않은 모델을 사용하여 Planning 메커니즘을 시연합니다. 실제 학습된 모델을 사용하면 성능이 크게 향상됩니다.
