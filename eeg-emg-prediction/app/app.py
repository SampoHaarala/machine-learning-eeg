"""Run from project root: python -m streamlit run app/app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from src.config import load_config, validate
from src.data.loader import files, iter_trials, inspect_file, load_npz, load_grasping_box
from src.data.dataset_builder import build_dataset, Dataset
from src.data.visualize import plot_trial
from src.data.events import align, read_annotations
from src.preprocessing.windows import extract
from src.training.train import train, output_guard
from src.training.evaluate import compare_runs
from src.storage.model_io import load_model, save_model
from src.inference.predict import EEGSample

st.set_page_config(page_title='EEG → EMG laboratory', layout='wide')
st.title('EEG → future muscle activation')
st.caption('Offline regression experiments • subject-separated evaluation • saved preprocessing and models')
if 'config' not in st.session_state:
    st.session_state.config = load_config('configs/default.yaml')
if 'executor' not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)
    st.session_state.messages = queue.Queue()
    st.session_state.stop = threading.Event()
c = copy.deepcopy(st.session_state.config)
with st.sidebar:
    st.subheader('Configuration')
    uploaded = st.file_uploader('Load YAML configuration', type=['yaml', 'yml'])
    if st.button('Apply uploaded configuration') and uploaded:
        try:
            st.session_state.config = validate(yaml.safe_load(uploaded.getvalue()))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    c['seed'] = st.number_input('Random seed', 0, 2**31-1, int(c['seed']))
    st.info('Use configs/demo.yaml after running the synthetic demo, or supply downloaded MATLAB data.')

sections = st.tabs(['Data', 'EEG', 'EMG target', 'Features', 'Model & split', 'Training', 'Results', 'Saved models', 'Prediction'])
with sections[0]:
    c['data']['path'] = st.text_input('Dataset directory', c['data']['path'])
    c['data']['format'] = st.selectbox('Dataset format', ['grasping_box', 'npz'], index=['grasping_box','npz'].index(c['data']['format']))
    c['data']['pattern'] = st.text_input('File pattern', c['data']['pattern'])
    left, right = st.columns(2)
    with left:
        c['data']['eeg_unit'] = st.selectbox('Source EEG units', ['V','mV','uV'], index=['V','mV','uV'].index(c['data']['eeg_unit']))
    with right:
        c['data']['emg_unit'] = st.selectbox('Source EMG units', ['V','mV','uV'], index=['V','mV','uV'].index(c['data']['emg_unit']))
    c['data']['units_confirmed'] = st.checkbox('I verified these units against the source files/documentation', c['data']['units_confirmed'])
    available = files(c)
    st.write(f'{len(available)} input files')
    if available:
        chosen = st.selectbox('File to inspect', [str(p) for p in available])
        if st.button('Inspect MATLAB/NPZ structure'):
            try:
                st.json(inspect_file(chosen))
            except Exception as exc:
                st.error(str(exc))
    if st.button('Read subject/trial catalog'):
        try:
            cc = copy.deepcopy(c)
            cc['data'].update(subjects=[], trials=[], tasks=[])
            with st.spinner('Reading recordings'):
                st.session_state.catalog = [{'subject': t.subject, 'trial_id': t.trial_id, 'task': t.task,
                    'block': t.block, 'fs': t.fs, 'eeg_channels': len(t.eeg_names), 'emg_channels': len(t.emg_names),
                    'source': t.source} for t in iter_trials(cc)]
        except Exception as exc:
            st.error(str(exc))
    catalog = st.session_state.get('catalog', [])
    if catalog:
        frame = pd.DataFrame(catalog)
        for key, column, title in [('subjects','subject','Subjects'), ('trials','trial_id','Trial IDs'), ('tasks','task','Grip types')]:
            options = sorted(frame[column].unique().tolist())
            c['data'][key] = st.multiselect(title+' (empty = all)', options, default=[str(x) for x in c['data'][key] if str(x) in options])
        st.dataframe(frame.groupby(['subject','task']).size().rename('trials').reset_index(), hide_index=True)
    if st.button('Visualize first selected trial'):
        try:
            st.pyplot(plot_trial(next(iter_trials(c)), c))
        except Exception as exc:
            st.error(str(exc))

with sections[1]:
    c['eeg']['bandpass'][0] = st.number_input('EEG low cutoff (Hz)', .01, value=float(c['eeg']['bandpass'][0]))
    c['eeg']['bandpass'][1] = st.number_input('EEG high cutoff (Hz)', 1., value=float(c['eeg']['bandpass'][1]))
    c['eeg']['notch'] = st.checkbox('EEG notch enabled', c['eeg']['notch'])
    c['eeg']['artifact_uv'] = st.number_input('Artifact threshold (µV)', 1., value=float(c['eeg']['artifact_uv']))
    c['eeg']['window'][0] = st.number_input('EEG window start (s)', value=float(c['eeg']['window'][0]))
    c['eeg']['window'][1] = st.number_input('EEG window end (s)', value=float(c['eeg']['window'][1]))
    st.caption('Common average reference uses retained scalp channels. Filtering is confined to the selected pre-event window.')
with sections[2]:
    events = ['emg_onset','annotated_onset','cue','touch','lift','movement_onset']
    c['alignment']['event'] = st.selectbox('Alignment event', events, index=events.index(c['alignment']['event']))
    if c['alignment']['event'] == 'annotated_onset':
        c['alignment']['annotations_csv'] = st.text_input('Onset CSV path', c['alignment']['annotations_csv'] or '')
    if c['alignment']['event'] != 'annotated_onset':
        st.warning('EMG onset is a derived offline proxy. Cue/touch/lift are different events, not movement onset. movement_onset is available in the synthetic fixture.')
    c['emg']['window'][0] = st.number_input('EMG target start (s)', value=float(c['emg']['window'][0]))
    c['emg']['window'][1] = st.number_input('EMG target end (s)', value=float(c['emg']['window'][1]))
    c['emg']['bandpass'][0] = st.number_input('EMG low cutoff (Hz)', .01, value=float(c['emg']['bandpass'][0]))
    c['emg']['bandpass'][1] = st.number_input('EMG high cutoff (Hz)', 1., value=float(c['emg']['bandpass'][1]))
    c['emg']['notch'] = st.checkbox('EMG notch enabled', c['emg']['notch'])
    targets = ['rms','mean_envelope']
    c['emg']['target'] = st.selectbox('Target calculation', targets, index=targets.index(c['emg']['target']))
    methods = ['balanced_subject_p95','robust','none','subject_p95']
    c['emg']['normalization'] = st.selectbox('Training-only target normalization', methods, index=methods.index(c['emg']['normalization']))
    st.caption('subject_p95 requires within-subject training calibration. Unseen-subject contact differences remain a limitation.')
with sections[3]:
    bands = st.text_area('Bands (YAML)', yaml.safe_dump(c['features']['bands'], sort_keys=False))
    channels = st.text_input('Selected EEG channels (comma separated; blank = all)', ','.join(c['eeg']['selected_channels']))
    c['eeg']['selected_channels'] = [s.strip() for s in channels.split(',') if s.strip()]
    c['features']['welch_seconds'] = st.number_input('Welch segment (seconds)', .05, value=float(c['features']['welch_seconds']))
    c['features']['overlap'] = st.slider('Welch overlap', 0., .95, float(c['features']['overlap']))
    st.caption('A 0.9-second epoch provides limited low-frequency resolution; zero-padding cannot recover missing information.')
with sections[4]:
    models = ['ridge', 'random_forest', 'mlp']
    c['model']['name'] = st.selectbox('Regression model', models, index=models.index(c['model']['name']))
    kind = c['model']['name']
    model_yaml = st.text_area('Model-specific hyperparameters (YAML)', yaml.safe_dump(c['model'][kind], sort_keys=False), key='params_'+kind)
    modes = ['subject','within_subject']
    c['split']['mode'] = st.selectbox('Evaluation design', modes, index=modes.index(c['split']['mode']))
    split_yaml = st.text_area('Split configuration (YAML)', yaml.safe_dump({k:v for k,v in c['split'].items() if k!='mode'}, sort_keys=False))
    st.caption('Subject counts must match the selected subjects. Use explicit subject lists to freeze an allocation.')

try:
    c['features']['bands'] = yaml.safe_load(bands)
    c['model'][kind] = yaml.safe_load(model_yaml)
    c['split'].update(yaml.safe_load(split_yaml))
    validate(c)
    valid = True
except Exception as exc:
    valid = False
    st.error(f'Configuration: {exc}')

with sections[5]:
    c['paths']['processed'] = st.text_input('Processed dataset path', c['paths']['processed'])
    experiment = st.text_input('Experiment name (blank = timestamp)', c['experiment']['name'] or '')
    c['experiment']['name'] = experiment or None
    future = st.session_state.get('future')
    busy = future is not None and not future.done()
    def launch(fn):
        st.session_state.stop.clear()
        st.session_state.future = st.session_state.executor.submit(fn)
        st.rerun()
    def build_job(cfg, messages, stop):
        def progress(message):
            if stop.is_set(): raise InterruptedError('Build cancelled')
            messages.put(message)
        output_guard(cfg, cfg['paths']['processed'])
        ds = build_dataset(cfg, progress)
        ds.save(cfg['paths']['processed'])
        return {'stage': 'built', 'samples': len(ds.X), 'path': cfg['paths']['processed']}
    def train_job(cfg, name, messages, stop):
        ds = Dataset.load(cfg['paths']['processed'])
        out, _, _ = train(ds, cfg, name or None, messages.put, stop.is_set)
        return {'stage': 'trained', 'run': str(out)}
    b1, b2, b3 = st.columns(3)
    if b1.button('Build dataset', disabled=busy or not valid):
        launch(lambda cfg=copy.deepcopy(c), q=st.session_state.messages, stop=st.session_state.stop: build_job(cfg,q,stop))
    if b2.button('Train and save experiment', disabled=busy or not valid):
        launch(lambda cfg=copy.deepcopy(c), name=experiment, q=st.session_state.messages, stop=st.session_state.stop: train_job(cfg,name,q,stop))
    if b3.button('Stop', disabled=not busy):
        st.session_state.stop.set()
    st.caption('Stop is checked between trials, model candidates, or MLP epochs. One forest fit finishes before cancellation.')
    @st.fragment(run_every='1s')
    def job_status():
        q = st.session_state.messages
        while not q.empty():
            st.session_state.last_progress = q.get()
        if 'last_progress' in st.session_state:
            message = st.session_state.last_progress
            st.json(message)
            if 'epoch' in message:
                st.progress(min(1., message['epoch']/max(1,c['model']['mlp']['max_epochs'])))
        f = st.session_state.get('future')
        if f is not None and f.done():
            try:
                result = f.result()
                st.session_state.last_job = result
            except Exception as exc:
                st.session_state.last_job = {'error': str(exc)}
            st.session_state.future = None
            st.rerun()
        if 'last_job' in st.session_state:
            st.write(st.session_state.last_job)
    job_status()
    if st.button('Reset status', disabled=busy):
        st.session_state.pop('last_progress', None)
        st.session_state.pop('last_job', None)
        st.rerun()

with sections[6]:
    runs = sorted(Path(c['paths']['runs']).glob('*/metrics.json'))
    if runs:
        chosen_run = st.selectbox('Experiment results', [str(p.parent) for p in runs])
        info = json.loads((Path(chosen_run)/'metrics.json').read_text())
        if info['synthetic']: st.warning('SYNTHETIC DATA — software validation only')
        which = st.radio('Evaluation split', ['validation','test'], horizontal=True)
        if which == 'test': st.caption('Repeated model selection on these test results invalidates the held-out claim.')
        st.dataframe(pd.DataFrame(info[which]['per_muscle']), hide_index=True)
        st.json(info[which]['macro'])
        with st.expander('Mean and grip-only baselines'):
            st.json({k:info[k] for k in (which+'_mean_baseline',which+'_grip_baseline')})
        for plot in sorted((Path(chosen_run)/'plots').glob('*.png')):
            st.image(str(plot), caption='Test-set plot: '+plot.stem)
        compare = st.multiselect('Compare validation results', [str(p.parent) for p in runs])
        if st.button('Generate model comparison') and compare:
            try:
                st.dataframe(compare_runs(compare, Path(c['paths']['runs'])/'comparison'))
                st.image(str(Path(c['paths']['runs'])/'comparison/model_comparison.png'))
            except Exception as exc: st.error(str(exc))
    else: st.info('Completed experiments appear here.')

with sections[7]:
    saved = sorted(Path(c['paths']['models']).glob('*/metadata.json'))
    if saved:
        selected_model = st.selectbox('Saved model', [str(p.parent) for p in saved])
        st.json(json.loads((Path(selected_model)/'metadata.json').read_text()))
        if st.button('Load selected model'):
            st.session_state.loaded_model = load_model(selected_model)
        extra = st.text_input('Save loaded model to a new directory', 'models/trained/copy')
        if st.button('Save loaded model') and 'loaded_model' in st.session_state:
            try:
                output_guard(c, extra)
                save_model(st.session_state.loaded_model, extra)
                st.success('Model saved')
            except Exception as exc: st.error(str(exc))
    st.caption('Only load trusted local model bundles; joblib is not safe for untrusted uploads.')

with sections[8]:
    model = st.session_state.get('loaded_model')
    if model is None:
        st.info('Load a saved model first.')
    else:
        st.write('Expected EEG channels:', model.schema['eeg_names'])
        st.write('Window:', model.config['eeg']['window'], 'Sampling rate:', model.schema['fs'])
        sample_upload = st.file_uploader('EEG sample NPZ (eeg, fs, channel_names, window)', type=['npz'])
        if sample_upload and st.button('Predict uploaded EEG'):
            try:
                with np.load(sample_upload, allow_pickle=False) as z:
                    sample = EEGSample(z['eeg'], float(z['fs']), z['channel_names'].tolist(), tuple(z['window']))
                values = model.predict(sample)
                st.bar_chart(pd.Series(values,index=model.muscle_names,name='Predicted normalized activation'))
                st.dataframe(pd.DataFrame({'muscle':model.muscle_names,'activation':values}))
            except Exception as exc: st.error(str(exc))
        if catalog:
            labels = [f'{r["subject"]}/{r["trial_id"]} · {r["task"]}' for r in catalog]
            selected_trial = st.selectbox('Or use one recorded trial', range(len(catalog)), format_func=lambda i:labels[i])
            if st.button('Predict selected recorded trial'):
                try:
                    row = catalog[selected_trial]
                    adapter = load_npz if Path(row['source']).suffix=='.npz' else load_grasping_box
                    trial = next(t for t in adapter(row['source'],model.config) if t.trial_id==row['trial_id'])
                    onset, _ = align(trial, model.config['alignment'], read_annotations(model.config['alignment']['annotations_csv']))
                    epoch, _ = extract(trial.eeg,onset,model.config['eeg']['window'],trial.fs)
                    sample = EEGSample(epoch,trial.fs,trial.eeg_names,tuple(model.config['eeg']['window']))
                    st.dataframe(pd.DataFrame({'muscle':model.muscle_names,'activation':model.predict(sample)}))
                except Exception as exc: st.error(str(exc))

with st.sidebar:
    st.download_button('Download current YAML', yaml.safe_dump(c, sort_keys=False), 'experiment.yaml')
    with st.expander('Full effective configuration'):
        st.json(c)
