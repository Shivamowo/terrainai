# TerrainAI Semantic Segmentation Baseline

This project implements a baseline SegFormer-B2 model for terrain segmentation using synthetic desert terrain data.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the dataset is downloaded and structured as:
   ```
   data/
     train/images/   # RGB .png images
     train/masks/    # Grayscale .png masks
     val/images/
     val/masks/
     testImages/     # No masks, for submission
   ```

## Usage

### Training

Run the baseline training:
```bash
python src/train.py
```

This will train for 10 epochs and save the best model to `checkpoints/run1_best.pth`.

### Debug Mode

For local testing on CPU with a small subset:
```bash
python src/train.py --debug
```

This runs 1 epoch on 10 images only.

## Project Structure

- `src/dataset.py`: Dataset class and mask remapping
- `src/utils.py`: IoU computation utilities
- `src/train.py`: Training script
- `checkpoints/`: Saved model checkpoints
- `data/`: Dataset directory

## Notes

- Masks are remapped from raw values to 0-9 classes
- Images are resized to 512x512 and normalized with ImageNet stats
- No augmentations in this baseline
- Always test with `--debug` before pushing code