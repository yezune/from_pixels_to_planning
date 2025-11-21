# VAE Training for Atari

This directory contains scripts for training and evaluating VAE models on Atari game observations.

## Overview

The VAE (Variational Autoencoder) learns to compress 64x64 RGB Atari frames into a compact 32-dimensional latent space. This is the first step in the Active Inference agent training pipeline described in the paper "From Pixels to Planning: Scale-Free Active Inference".

## Quick Start

### 1. Train VAE

Train a VAE on Breakout for 100 episodes and 100 epochs:

```bash
python src/experiments/train_atari_vae.py \
  --env_name Breakout \
  --num_episodes 100 \
  --epochs 100 \
  --batch_size 128 \
  --output_dir outputs/vae_training
```

**Quick test** (5 episodes, 10 epochs, ~30 seconds):
```bash
python src/experiments/train_atari_vae.py \
  --env_name Breakout \
  --num_episodes 5 \
  --epochs 10 \
  --batch_size 32 \
  --output_dir outputs/vae_quick_test
```

### 2. Evaluate Trained Model

Evaluate reconstruction quality:

```bash
python src/experiments/evaluate_vae.py \
  --model_path outputs/vae_training/best_model.pt \
  --env_name Breakout \
  --num_samples 1000 \
  --output_dir outputs/vae_evaluation
```

## Training Arguments

### `train_atari_vae.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--env_name` | `Breakout` | Atari environment name (e.g., Breakout, Pong, SpaceInvaders) |
| `--latent_dim` | `32` | Dimensionality of latent space |
| `--num_episodes` | `100` | Number of random episodes to collect |
| `--max_steps` | `1000` | Maximum steps per episode |
| `--batch_size` | `128` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--val_split` | `0.1` | Validation split ratio |
| `--device` | `auto` | Device (cpu/cuda/mps/auto) |
| `--output_dir` | `outputs/vae_training` | Output directory for checkpoints and visualizations |
| `--checkpoint_freq` | `10` | Save checkpoint every N epochs |
| `--seed` | `42` | Random seed |

## Training Output

The training script produces:

1. **Model Checkpoints**
   - `best_model.pt` - Best model based on validation loss
   - `final_model.pt` - Model after last epoch
   - `checkpoint_epoch_N.pt` - Periodic checkpoints with full training state

2. **Visualizations**
   - `training_curves.png` - Loss curves (total, reconstruction, KL divergence)
   - `reconstructions_epoch_N.png` - Original vs reconstructed frames
   - `final_reconstructions.png` - Final reconstructions

## Evaluation Metrics

The evaluation script computes:

1. **MSE** - Mean Squared Error between original and reconstructed pixels
2. **PSNR** - Peak Signal-to-Noise Ratio (higher is better, >30 dB is good)
3. **Accuracy** - Percentage of pixels within ±0.1 of original

### Example Results

After training for 100 episodes and 100 epochs on Breakout:
- **PSNR**: 30-35 dB
- **Accuracy**: 98-99%
- **Training time**: ~10-15 minutes (on Apple M-series GPU)

Quick test (5 episodes, 10 epochs):
- **PSNR**: ~31 dB
- **Accuracy**: ~99%
- **Training time**: ~30 seconds

## Training Tips

### Data Collection

- **Random episodes** provide diverse observations
- **100 episodes** gives ~10K-50K frames (enough for initial training)
- For production training, consider:
  - 1000+ episodes for better coverage
  - Mixed policies (random + partially trained agent)
  - Multiple Atari games for generalization

### Training

- Start with default hyperparameters
- If reconstruction quality is poor:
  - Increase `--epochs` to 200+
  - Increase `--num_episodes` to 500+
  - Try higher `--latent_dim` (64 or 128)
- If KL divergence dominates loss:
  - Reduce KL weight in VAE loss function
  - This is controlled in `src/models/base_vae.py`

### Evaluation

- Always evaluate on fresh samples (not from training set)
- Test on multiple Atari games to check generalization
- PSNR > 30 dB indicates good reconstruction quality
- Latent space should be smooth and continuous

## Integration with Planning

Once VAE is trained, it can be used with the planning module:

```python
from src.models.vae import VAE
from src.planning.mcts import MCTS

# Load trained VAE
vae = VAE(input_shape=(3, 64, 64), latent_dim=32)
vae.load_state_dict(torch.load('outputs/vae_training/best_model.pt'))
vae.eval()

# Use with MCTS planning
# (Planning code uses VAE to encode observations)
```

## File Structure

```
src/experiments/
├── train_atari_vae.py      # Main training script
├── evaluate_vae.py          # Evaluation and visualization
└── README_VAE_TRAINING.md   # This file

outputs/
├── vae_training/            # Full training run
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── checkpoint_epoch_*.pt
│   ├── training_curves.png
│   └── reconstructions_*.png
└── vae_evaluation/          # Evaluation results
    ├── metrics.txt
    ├── reconstructions.png
    ├── latent_space.png
    └── prior_samples.png
```

## Next Steps

After training VAE:

1. **Train Transition Model** - Learn temporal dynamics in latent space
2. **Train Policy Network** - Learn to plan using MCTS with learned models
3. **Full Active Inference Training** - Combine all components

See `src/hierarchical_trainer.py` for the full training pipeline.

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` to 64 or 32
- Reduce `--num_episodes` to collect fewer frames

### Poor Reconstruction Quality
- Increase `--epochs` to 200+
- Increase `--num_episodes` to 500+
- Check if latent_dim is too small (try 64 or 128)

### Training Too Slow
- Use GPU (CUDA or MPS will be auto-detected)
- Reduce `--num_episodes` for quick iterations
- Use `--checkpoint_freq` to save less frequently

### Import Errors
- Make sure you're running from project root
- Activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
