import os
import torch
from models.model import get_model

CHECKPOINT_CANDIDATES = ["cnn_updated.pth", "best_model.pth"]


def find_checkpoint_path():
    for candidate in CHECKPOINT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def normalize_state_dict(state):
    normalized = {}
    for key, value in state.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def infer_model_type(state_dict):
    for key in state_dict.keys():
        if key.startswith("resnet."):
            return "resnet"
    return "custom"


def load_model_from_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = normalize_state_dict(state_dict)
    model_type = infer_model_type(state_dict)
    model_kwargs = {"num_classes": 2}
    if model_type == "resnet":
        model_kwargs["pretrained"] = False
    model = get_model(model_type=model_type, **model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=False)
    accuracy = None
    if isinstance(checkpoint, dict):
        accuracy = checkpoint.get("val_acc") or checkpoint.get("best_val_acc")
    return model, model_type, accuracy


def test_model():
    print("Handwriting Forgery Detection Test")
    print("=" * 40)

    checkpoint_path = find_checkpoint_path()
    if not checkpoint_path:
        print("No checkpoint found. Place 'cnn_updated.pth' or 'best_model.pth' in the project root.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model, model_type, accuracy = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    print(f"Model architecture detected: {model_type}")
    if accuracy is not None:
        print(f"Validation accuracy stored in checkpoint: {accuracy:.2f}%")

    sample_input = torch.randn(1, 1, 224, 224).to(device)
    with torch.no_grad():
        output = model(sample_input)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100

    class_names = ["Genuine", "Forged"]
    print("\nTesting model with random sample...")
    print(f"Sample prediction: {class_names[prediction]}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Genuine probability: {probabilities[0][0].item()*100:.1f}%")
    print(f"Forged probability: {probabilities[0][1].item()*100:.1f}%")
    print("\nModel test completed successfully!")


if __name__ == "__main__":
    test_model()













