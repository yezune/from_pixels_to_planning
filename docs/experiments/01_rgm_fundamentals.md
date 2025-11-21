# Experiment 1: RGM Fundamentals (Visualizing Renormalization)

## 1. 이론적 배경 (Theoretical Background)

이 실험은 **Renormalizing Generative Model (RGM)**의 핵심 원리인 **"Renormalization (재규격화)"** 과정을 시각적으로 증명하기 위한 독립적인 실험입니다. 복잡한 강화학습 환경 대신 직관적인 MNIST 데이터셋을 사용하여, RGM이 어떻게 세상을 계층적으로 이해하는지 보여줍니다.

### 핵심 가설

> "세상은 계층적(Hierarchical)이며, 국소적인 세부 사항(Local Details)은 상위 레벨에서 추상적인 개념(Abstract Concepts)으로 통합(Renormalized)된다."

이 실험은 다음 세 가지 특성을 검증합니다:

1. **Abstraction (추상화)**: 픽셀 $\to$ 국소 특징($z_1$) $\to$ 전역 개념($z_2$)으로 정보가 압축되는 과정.
2. **Instantiation (구체화)**: 하나의 개념($z_2$)이 다양한 형태($z_1$)로 발현되는 과정 (One Concept, Many Variations).
3. **Locality (국소성)**: 하위 레벨의 변화는 국소적이지만, 상위 레벨의 변화는 전역적인 영향을 미침.

## 2. 실험 설정 (Setup)

* **데이터셋 (Dataset)**: MNIST (28x28 Grayscale Images).
* **모델 (Model)**: Spatial RGM (2-Level Hierarchical Model).
  * **Level 1 ($z_1$)**: 7x7 Grid of Categorical Variables (Local Features).
  * **Level 2 ($z_2$)**: Global Categorical Variable (Digit Class).

## 3. 실험 과정 (Procedure)

### Experiment 1: Hierarchical Abstraction (Bottom-Up)

이미지가 입력되었을 때, 모델 내부의 각 계층이 무엇을 표현하는지 시각화합니다.

* **분석 대상**:
  * $z_1$ (7x7 Grid): 이미지의 각 부분이 어떤 특징(직선, 곡선 등)으로 인코딩되는지 확인.
  * $z_2$ (Class): 최종적으로 모델이 이미지를 어떤 숫자로 인식했는지 확인.

### Experiment 2: Concept-Conditional Generation (Top-Down)

상위 개념을 고정하고 하위 세부 사항을 변화시켜 생성 모델의 다양성을 확인합니다.

* **과정**:
  1. $z_2$를 특정 숫자(예: '3')로 고정.
  2. $P(z_1 | z_2)$ 분포에서 $z_1$을 여러 번 샘플링.
  3. 생성된 이미지들이 모두 '3'이면서도 서로 다른 필체를 가지는지 확인.

### Experiment 3: Local vs Global Perturbation

잠재 변수(Latent Variable)를 인위적으로 조작(Perturbation)하여 그 영향을 관찰합니다.

* **Local Perturbation**: $z_1$ 그리드의 한 셀만 변경 $\to$ 이미지의 해당 위치만 변해야 함.
* **Global Perturbation**: $z_2$ 클래스를 변경 $\to$ 이미지 전체가 다른 숫자로 변해야 함.

## 4. 실행 방법 (Execution)

이 실험은 Jupyter Notebook으로 구성되어 있습니다.

```bash
# 노트북 실행
jupyter notebook notebooks/05_rgm_independent_experiment.ipynb
```

또는 테스트 스크립트를 통해 실행 여부를 검증할 수 있습니다.

```bash
python3 -m unittest tests/test_notebooks.py
```

## 5. 예상 결과 (Expected Results)

* **Abstraction**: $z_1$ 시각화 결과, 숫자의 형태를 따라 의미 있는 패턴이 7x7 그리드에 나타납니다.
* **Generation**: 동일한 숫자 클래스 내에서도 다양한 스타일의 이미지가 생성됩니다.
* **Locality**: $z_1$의 변화는 픽셀 공간에서 국소적인 변화로 이어지며, 이는 RGM이 공간적 위상(Topology)을 잘 보존하고 있음을 증명합니다.
