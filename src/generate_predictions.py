"""
TerrainAI — leaderboard submission script.
Generates raw class-index masks (values 0-9) for every image in data/testImages/.
Output is written to predictions/ with identical filenames.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import load_model, preprocess_frame, predict_frame


def main():
    parser = argparse.ArgumentParser(description='Generate TerrainAI leaderboard predictions')
    parser.add_argument('--root', default=str(Path(__file__).parent.parent),
                        help='Project root directory')
    args = parser.parse_args()

    root = Path(args.root)
    checkpoint = root / 'checkpoints' / 'run_best.pth'
    # Images live in testImages/Images/ subdirectory
    test_dir = root / 'data' / 'testImages' / 'Images'
    pred_dir = root / 'predictions'

    if not checkpoint.exists():
        print(f'ERROR: checkpoint not found at {checkpoint}')
        sys.exit(1)
    if not test_dir.exists():
        print(f'ERROR: testImages directory not found at {test_dir}')
        sys.exit(1)

    pred_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model from {checkpoint} on {device}...')
    model = load_model(str(checkpoint), device)

    image_files = sorted(test_dir.glob('*.png'))
    if not image_files:
        print(f'No .png files found in {test_dir}')
        print(f'Processed: 0 | In predictions/: 0 | Match: True')
        return

    print(f'Found {len(image_files)} images. Running inference...')
    processed = 0

    for img_path in tqdm(image_files, desc='Predicting', unit='img'):
        import cv2
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f'WARNING: Could not read {img_path.name}, skipping.')
            continue

        h, w = frame.shape[:2]
        tensor = preprocess_frame(frame)
        mask = predict_frame(model, tensor, device, h, w)  # (H,W) uint8, values 0-9

        out_path = pred_dir / img_path.name
        Image.fromarray(mask, mode='L').save(str(out_path))
        processed += 1

    in_predictions = len(list(pred_dir.glob('*.png')))
    match = processed == len(image_files) == in_predictions

    print(f'\nProcessed:        {processed}')
    print(f'In predictions/:  {in_predictions}')
    print(f'Match:            {match}')

    # Sample check
    sample_files = list(pred_dir.glob('*.png'))
    if sample_files:
        sample = np.array(Image.open(sample_files[0]))
        print(f'\nSample prediction: {sample_files[0].name}')
        print(f'  dtype:         {sample.dtype}')
        print(f'  shape:         {sample.shape}')
        print(f'  unique values: {np.unique(sample).tolist()}')
        print(f'  max value:     {sample.max()}')
        if sample.max() <= 9:
            print('Value range OK (0-9)')
        else:
            print('WARNING: values out of expected range 0-9')


if __name__ == '__main__':
    main()
