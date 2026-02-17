from typing import Tuple

import numpy as np
import pyworld as pw

from .base import ThresholdPitchAlgorithm


class WORLDDioPitchAlgorithm(ThresholdPitchAlgorithm):
    _name = "WORLD-DIO"

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio_f64 = audio.astype(np.float64)
        frame_period_ms = 1000.0 * self.hop_size / self.sample_rate

        f0, temporal_positions = pw.dio(
            audio_f64,
            self.sample_rate,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=frame_period_ms,
            allowed_range=threshold,
        )
        f0 = pw.stonemask(audio_f64, f0, temporal_positions, self.sample_rate)

        periodicity = (f0 > 0).astype(np.float32)
        return temporal_positions, f0, periodicity

    def _get_default_threshold(self) -> float:
        return 0.1


class WORLDHarvestPitchAlgorithm(ThresholdPitchAlgorithm):
    _name = "WORLD-Harvest"

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, _threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio_f64 = audio.astype(np.float64)
        frame_period_ms = 1000.0 * self.hop_size / self.sample_rate

        f0, temporal_positions = pw.harvest(
            audio_f64,
            self.sample_rate,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=frame_period_ms,
        )
        f0 = pw.stonemask(audio_f64, f0, temporal_positions, self.sample_rate)

        periodicity = (f0 > 0).astype(np.float32)
        return temporal_positions, f0, periodicity

    def _get_default_threshold(self) -> float:
        return 0.1
