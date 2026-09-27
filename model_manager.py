# model_manager.py
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class ModelManager:
    """Simple utility to save and reload trained ML models as .pkl files."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save a model object to a pickle file."""
        model_path = self.models_dir / f"{model_name}.pkl"
        temp_path = self.models_dir / f"{model_name}.tmp"

        payload = {
            "model": model,
            "metadata": metadata or {},
        }

        with open(temp_path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temp_path, model_path)
        return model_path

    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load a saved model artifact.

        Returns:
            A dictionary containing 'model' and 'metadata' if found,
            otherwise None.
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            return None

        with open(model_path, "rb") as file:
            return pickle.load(file)

    def model_exists(self, model_name: str) -> bool:
        return (self.models_dir / f"{model_name}.pkl").exists()

    def delete_model(self, model_name: str) -> bool:
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            return False
        model_path.unlink()
        return True

    def list_models(self):
        return sorted(p.stem for p in self.models_dir.glob("*.pkl"))
