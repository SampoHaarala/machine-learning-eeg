"""The acquisition-independent model interface accepts raw, windowed EEG in volts."""
from dataclasses import dataclass
import numpy as np
from src.preprocessing.eeg import preprocess_eeg
from src.features.eeg_features import extract_features


@dataclass
class EEGSample:
    data: np.ndarray
    fs: float
    channel_names: list
    window: tuple


class PredictionModel:
    def __init__(self, estimator, target_scaler, config, schema, feature_names):
        self.estimator = estimator
        self.target_scaler = target_scaler
        self.config = config
        self.schema = schema
        self.feature_names = feature_names
        self.muscle_names = schema['emg_names']

    def features(self, epoch):
        if isinstance(epoch, EEGSample):
            if not np.isclose(epoch.fs, self.schema['fs']):
                raise ValueError('Sampling rate mismatch; resampling must be explicit')
            if epoch.channel_names != self.schema['eeg_names']:
                raise ValueError('EEG channel order differs from training')
            if not np.allclose(epoch.window, self.config['eeg']['window']):
                raise ValueError('EEG time window differs from training')
            x = epoch.data
        else:
            # A bare ndarray explicitly adopts the saved fs/order/window/volts contract.
            x = np.asarray(epoch)
        n = round(np.diff(self.config['eeg']['window'])[0] * self.schema['fs'])
        if x.shape != (len(self.schema['eeg_names']), n):
            raise ValueError(f'Expected raw EEG shape {(len(self.schema["eeg_names"]), n)} in volts; got {x.shape}')
        clean, names = preprocess_eeg(x, self.schema['fs'], self.schema['eeg_names'], self.config['eeg'])
        values, features = extract_features(clean, self.schema['fs'], names, self.config['features'])
        if features != self.feature_names:
            raise ValueError('Feature order mismatch')
        return values[None, :]

    def predict(self, eeg_epoch, *, units='normalized', subject=None):
        y = self.estimator.predict(self.features(eeg_epoch))
        if units == 'V':
            y = self.target_scaler.inverse_transform(y, [subject] if subject is not None else None)
        elif units != 'normalized':
            raise ValueError('Output units must be normalized or V')
        # Linear regression can produce negatives; preserve them for honest evaluation.
        return y[0]


def predict(model, eeg_epoch, **kwargs):
    return model.predict(eeg_epoch, **kwargs)
