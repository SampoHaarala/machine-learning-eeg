import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.data.events import align, read_annotations, EventError


def plot_trial(t, c):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    time = np.arange(t.eeg.shape[1])/t.fs
    # A small selection is legible; preserve all raw channels in the loader.
    for i in range(min(4, len(t.eeg_names))):
        ax[0].plot(time, t.eeg[i]*1e6 + i*70, lw=.5, label=t.eeg_names[i])
    for i in range(min(4, len(t.emg_names))):
        ax[1].plot(time, t.emg[i]*1e3 + i, lw=.5, label=t.emg_names[i])
    for name, event in t.events.items():
        for a in ax:
            a.axvline(event, color='gray', ls=':', alpha=.7)
        ax[0].text(event, ax[0].get_ylim()[1], name, fontsize=8)
    try:
        onset, _ = align(t, c['alignment'], read_annotations(c['alignment']['annotations_csv']))
        for a in ax:
            a.axvline(onset, color='black', label=c['alignment']['event'])
        ax[0].axvspan(onset+c['eeg']['window'][0], onset+c['eeg']['window'][1], alpha=.2, color='blue')
        ax[1].axvspan(onset+c['emg']['window'][0], onset+c['emg']['window'][1], alpha=.2, color='orange')
    except EventError as exc:
        ax[0].set_title(str(exc))
    ax[0].set_ylabel('EEG µV, offset')
    ax[1].set_ylabel('EMG mV, offset')
    ax[1].set_xlabel('Seconds from trial start')
    for a in ax:
        a.legend(loc='upper right', fontsize=7)
    fig.suptitle(f'Subject {t.subject} · trial {t.trial_id} · {t.task}')
    fig.tight_layout()
    return fig
