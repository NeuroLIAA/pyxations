from abc import ABC, abstractmethod


class EyeMovementDetection(ABC):
    @abstractmethod
    def detect_eye_movements(self, *args, **kwargs):
        """Return detected eye-movement events."""
