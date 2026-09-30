import numpy as np
import torch
import torch.nn.functional as F

from train_inception_5class_recovered import InceptionTime


class InceptionPredictor:
    """
    Inference wrapper for the protected 5-class InceptionTime model.

    Input:
        X: numpy array of shape (N, 5000, 12)

    Output class order:
        [SB, AFIB, SR, ST, SVT]
    """

    CLASS_ORDER = [
        "Sinus Bradycardia",
        "Atrial Fibrillation",
        "Sinus Rhythm",
        "Sinus Tachycardia",
        "Supraventricular Tachycardia",
    ]

    # Native model order:
    # [AFIB, SB, SR, ST, SVT]
    # Avani/evaluator order:
    # [SB, AFIB, SR, ST, SVT]
    REORDER = [1, 0, 2, 3, 4]

    def __init__(
        self,
        checkpoint_path=r"models\best_inception_5class_PROTECTED.pth",
        device=None,
    ):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = InceptionTime(num_classes=5).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_proba(self, X, batch_size=64):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (N, 5000, 12), got {X.shape}"
            )

        if X.shape[1:] != (5000, 12):
            raise ValueError(
                f"Expected X with shape (N, 5000, 12), got {X.shape}"
            )

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(
                    X[start:start + batch_size]
                ).to(self.device)

                # (N, 5000, 12) -> (N, 12, 5000)
                batch = batch.transpose(1, 2)

                logits = self.model(batch)
                native_probs = F.softmax(logits, dim=1)

                # Native:
                # [AFIB, SB, SR, ST, SVT]
                #
                # Return:
                # [SB, AFIB, SR, ST, SVT]
                avani_probs = native_probs[:, self.REORDER]

                probabilities.append(
                    avani_probs.cpu().numpy()
                )

        return np.concatenate(probabilities, axis=0)


if __name__ == "__main__":
    print("Creating InceptionPredictor...")
    predictor = InceptionPredictor()

    print("Device:", predictor.device)
    print("Model loaded successfully.")
    print("Output classes:", predictor.CLASS_ORDER)
