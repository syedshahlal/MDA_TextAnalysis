import os
import torch
import pickle
from src.model import FraudDetectionMLP

# def get_model_path(num_features, model_dir="model"):
#     return os.path.join(model_dir, f"fraud_detection_model_{num_features}_features.pth")

def model_exists(num_features, model_dir="model"):
    model_path = f"{model_dir}/fraud_detection_model_{num_features}_features.pth"
    os.chdir(r'D:\University\UB\Research_SEC\MDA_TextAnalysis')

    return os.path.isfile(model_path)

def save_model(model, num_features, metadata, model_dir="model"):
    model_path = f"{model_dir}/fraud_detection_model_{num_features}_features.pth"
    metadata_path = f"{model_dir}/metadata_{num_features}_features.pkl"
    torch.save(model.state_dict(), model_path)
    with open(metadata_path, 'wb') as meta_handle:
        pickle.dump(metadata, meta_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model and metadata with {num_features} features saved to {model_dir}")


def load_model(num_features, model_dir="model"):
    model_path = f"{model_dir}/fraud_detection_model_{num_features}_features.pth"
    metadata_path = f"{model_dir}/metadata_{num_features}_features.pkl"
    model = FraudDetectionMLP(num_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with open(metadata_path, 'rb') as meta_handle:
        metadata = pickle.load(meta_handle)
    return model, metadata
