from abc import ABC, abstractmethod


class EyeMovementDetection(ABC):
    """Base class for eye-movement detection adapters.

    Subclasses wrap one detection algorithm and turn processed gaze samples
    into fixation, saccade and blink tables. Pyxations ships
    :class:`~pyxations.EngbertDetection` and
    :class:`~pyxations.RemodnavDetection`; EyeLink recordings can instead reuse
    the events reported by the vendor parser.

    Implement :meth:`detect_eye_movements` to add support for another
    algorithm. The canonical BIDS storage layer does not change, so a new
    detector becomes usable across the whole analysis hierarchy without
    touching the conversion or export code.
    """

    @abstractmethod
    def detect_eye_movements(self, *args, **kwargs):
        """Return detected eye-movement events.

        Parameters
        ----------
        *args : object
            Positional detector-specific configuration values.
        **kwargs : object
            Keyword detector-specific configuration values.
        """
