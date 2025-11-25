# L-AGI Pong Experiment

## Overview
This experiment applies the Logical Free Energy Principle (L-FEP) to the Atari Pong environment.
It uses a CNN-based agent (`LogicalPongAgent`) with Spherical Activation and Logical Divergence.

## Setup
- **Environment**: `PongNoFrameskip-v4` (via Gymnasium & Ale-py)
- **Agent**: `LogicalPongAgent`
    - Input: 4 stacked frames (grayscale, 64x64)
    - Architecture: 2 Conv layers -> FC -> Spherical Activation
    - Output: 6 actions (Pong default)
- **Training**: REINFORCE-style update with Intrinsic Motivation (Distinction Bonus).

## Initial Verification
- **Script**: `src/experiments/run_pong_l_agi.py`
- **Status**: Successfully ran 5 episodes on CPU.
- **Results**:
    - Episodes: 5
    - Mean Reward: -21.0 (Expected for random/untrained agent)
    - Loss: Fluctuating (Gradients are flowing)

## Next Steps
1. **Long-term Training**: Run for 1000+ episodes to observe learning.
2. **Hyperparameter Tuning**: Adjust learning rate, gamma, and intrinsic weight.
3. **Architecture**: Experiment with deeper networks or different frame stacking.
4. **Algorithm**: Implement PPO or A2C style updates for better stability than REINFORCE.
