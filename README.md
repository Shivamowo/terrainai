# TerrainAI Semantic Segmentation

This project implements SegFormer-B2 for terrain segmentation using synthetic desert terrain data.

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

Run the training:
```bash
cd C:\Users\avani\terrainai
conda activate terrainai
python src/train.py --root C:\Users\avani\terrainai
```

This will train for 20 epochs and save the best model to `checkpoints/run_best.pth`.

### Debug Mode

For local testing on CPU with a small subset:
```bash
python src/train.py --debug
```

This runs 5 epochs on 10 images only.

## Project Structure

- `src/dataset.py`: Dataset class with augmentation and mask remapping
- `src/model.py`: Model loading with SMP and HuggingFace fallback
- `src/train.py`: Training script with combined loss and logging
- `src/utils.py`: IoU computation and class names
- `src/augment.py`: Albumentations pipeline
- `checkpoints/`: Saved model checkpoints
- `logs/`: Training logs and results CSV
- `predictions/`: For inference outputs
- `data/`: Dataset directory

## Notes

- Masks are remapped from raw values to 0-9 classes
- Classes 8 (Logs) and 9 (Flowers) are upweighted in loss
- Combined CrossEntropy + Dice loss
- Always test with `--debug` before full training