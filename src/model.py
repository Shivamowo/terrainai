import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class ModelWrapper(nn.Module):
    def __init__(self, model, is_hf=False):
        super().__init__()
        self.model = model
        self.is_hf = is_hf

    def forward(self, x):
        outputs = self.model(x)
        if self.is_hf:
            return outputs.logits
        else:
            return outputs

def get_model(num_classes=10, device='cuda'):
    """
    Load SegFormer-B2 model. Try SMP first, fallback to HuggingFace with custom head.

    Args:
        num_classes (int): Number of classes.
        device (str): Device to move model to.

    Returns:
        nn.Module: The model.
    """
    try:
        model = smp.create_model('segformer_b2', encoder_weights='imagenet', classes=num_classes)
        wrapped_model = ModelWrapper(model, is_hf=False)
    except Exception as e:
        print(f"SMP failed: {e}. Falling back to HuggingFace.")
        # Load config and modify for our num_classes
        config = SegformerConfig.from_pretrained('nvidia/mit-b2')
        config.num_labels = num_classes
        config.id2label = {str(i): str(i) for i in range(num_classes)}
        config.label2id = {str(i): i for i in range(num_classes)}

        model = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/mit-b2',
            config=config,
            ignore_mismatched_sizes=True
        )
        wrapped_model = ModelWrapper(model, is_hf=True)

    wrapped_model.to(device)
    return wrapped_model