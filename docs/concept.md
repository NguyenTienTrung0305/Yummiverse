## Training Loop

- **num_epochs**: total number of repetitions => The larger → longer learning, risk of overfitting if too much
- **steps_per_epoch**: number of batch in 1 epoch => More → smaller and more frequent updates (more stable)
- **batch_size**: Number of images in 1 batch

- **total_steps**: Total number of updates in the entire training
- **num_samples**: Total number of images in dataset

## Optimizer & Scheduler

- **AdamW**: adaptive + momentum + decoupled weight decay.
- **OneCycleLR**: cyclic LR schedule (warm-up → peak → cooldown).

  - **pct_start**: Percentage (%) of training time spent in the “LR increase” (warm-up) phase
  - **anneal_strategy**: Strategy to reduce LR after peak (max_lr).
  - **div_factor**: used to calculate the initial LR value

- **Weight Decay**: shrinks large weights to prevent overfitting.
- **Dropout (0.3)**: randomly disables neurons to generalize better.
- **Data Augmentation**: random rotation, brightness, noise, etc

## Dataset Splitting

- **train/val**: 80/20 using StratifiedKFold (preserves class balance).
