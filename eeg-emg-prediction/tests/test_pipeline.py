import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat
from src.config import load_config
from src.data.loader import Trial, load_grasping_box, read_mat
from src.data.dataset_builder import sample, Dataset, build_dataset
from src.data.events import align, EventError
from src.data.synthetic import generate
from src.preprocessing.emg import TargetScaler
from src.preprocessing.eeg import ArtifactError
from src.preprocessing.windows import extract
from src.training.split import make_split
from src.training.train import train
from src.training.evaluate import metrics
from src.storage.model_io import load_model
from src.inference.predict import EEGSample
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def cfg():
    c = load_config(ROOT/'configs/default.yaml')
    c['alignment']['event'] = 'movement_onset'
    return c

@pytest.fixture
def trial():
    rng = np.random.default_rng(5)
    fs=1000.; t=np.arange(4000)/fs
    eeg=np.stack([np.sin(2*np.pi*(8+i)*t)*8e-6 for i in range(8)])
    emg=rng.normal(0,1e-6,(3,len(t)));emg[:,2000:3000] *= 100
    return Trial('1','1','PG','1',eeg,emg,fs,[f'E{i}' for i in range(8)],
                 ['M0','M1','M2'],{'cue':1.7,'movement_onset':2.},'synthetic',{'synthetic':True})

def test_no_future_eeg_leakage(cfg,trial):
    a=sample(trial,cfg);other=copy.deepcopy(trial)
    other.eeg[:,1900:]=1e4
    b=sample(other,cfg)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

def test_emg_never_features(cfg,trial):
    a=sample(trial,cfg);other=copy.deepcopy(trial);other.emg*=10;b=sample(other,cfg)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(b[1],10*a[1])

def test_target_confined_to_window(cfg,trial):
    a=sample(trial,cfg);other=copy.deepcopy(trial)
    other.emg[:,:2000]=100;other.emg[:,2500:]=100;b=sample(other,cfg)
    np.testing.assert_array_equal(a[1],b[1])

def test_onset_detector(cfg,trial):
    cfg['alignment']['event']='emg_onset'
    time,info=align(trial,cfg['alignment'])
    assert 2 <= time < 2.06
    assert info['method']=='emg_onset'
    trial.emg[:]=0
    with pytest.raises(EventError,match='no_emg_onset'): align(trial,cfg['alignment'])

def test_artifact_and_invalid_window(cfg,trial):
    trial.eeg[0,1200]=1
    with pytest.raises(ArtifactError): sample(trial,cfg)
    cfg['eeg']['window']=[-1,.1]
    from src.config import validate
    with pytest.raises(ValueError): validate(cfg)

def test_split_no_subject_leakage(cfg):
    metadata=[{'subject':str(s),'trial_id':str(t),'block':str(t//2)} for s in range(14) for t in range(6)]
    splits,report=make_split(metadata,cfg['split'],42)
    ss=[set(report[k]['subjects']) for k in report]
    assert list(map(len,ss))==[10,2,2]
    assert not (ss[0]&ss[1] or ss[0]&ss[2] or ss[1]&ss[2])
    assert sum(map(len,splits.values()))==len(metadata)
    assert report==make_split(metadata,cfg['split'],42)[1]
    cfg['split']['subjects']={'train':['0'],'validation':['0'],'test':['1']}
    with pytest.raises(ValueError): make_split(metadata,cfg['split'],42)

def test_within_subject_blocks(cfg):
    cfg['split']['mode']='within_subject'
    metadata=[{'subject':str(s),'trial_id':str(t),'block':str(t//2)} for s in range(2) for t in range(8)]
    split,_=make_split(metadata,cfg['split'],42)
    groups=[{(metadata[i]['subject'],metadata[i]['block']) for i in ids} for ids in split.values()]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

def test_training_only_target_scaling():
    y=np.array([[1.,2.],[3.,4.],[2.,3.],[4.,5.]])
    scaler=TargetScaler().fit(y,['1','1','2','2'])
    expected=np.median([np.percentile(y[:2],95,axis=0),np.percentile(y[2:],95,axis=0)],axis=0)
    np.testing.assert_allclose(scaler.scale_,expected)
    old=scaler.scale_.copy();scaler.transform(np.ones((2,2))*1e9,['new','new'])
    np.testing.assert_array_equal(old,scaler.scale_)
    subject=TargetScaler('subject_p95').fit(y,['1','1','2','2'])
    with pytest.raises(ValueError): subject.transform(y[:1],['new'])

def test_metrics_constants():
    result=metrics(np.ones((5,2)),np.ones((5,2)),['a','b'])
    assert result['macro']['MAE']==0
    assert result['per_muscle'][0]['R2'] is None
    assert result['per_muscle'][0]['Pearson_r'] is None
    json.dumps(result,allow_nan=False)

def test_published_mat_schema(cfg,trial,tmp_path):
    cfg['data'].update(units_confirmed=True,expected_eeg_channels=8,expected_emg_channels=3)
    data={'subject':{'id':1,'hand':'RIGHT'},'info':{'fs':1000.,
          'eeg_channels':np.array([{'labels':n} for n in trial.eeg_names],dtype=object),
          'emg_channels':np.array(trial.emg_names,dtype=object),'tasks_names':np.array(['PG','UG','WH'],dtype=object)},
          'trials':np.array([{'eeg':trial.eeg*1e6,'emg':trial.emg*1e3,'LEDon_event_frame':1701,
            'touch_event_frame':2601,'lift_event_frame':3001,'task':1,'block':1,'event_complete_trial':1}],dtype=object)}
    path=tmp_path/'s_1.mat';savemat(path,{'data':data})
    t=list(load_grasping_box(path,cfg))[0]
    assert t.events['cue']==1.7;assert t.eeg_names==trial.eeg_names
    np.testing.assert_allclose(t.eeg,trial.eeg);np.testing.assert_allclose(t.emg,trial.emg)

def test_v73_references(tmp_path):
    import h5py
    path=tmp_path/'v73.mat'
    with h5py.File(path,'w') as f:
        group=f.create_group('data');group.attrs['MATLAB_class']=np.bytes_('struct')
        ref=f.create_group('#refs#');x=ref.create_dataset('signal',data=np.arange(12).reshape(4,3))
        cell=group.create_dataset('eeg',shape=(1,1),dtype=h5py.ref_dtype);cell[0,0]=x.ref
        x=ref.create_dataset('label',data=np.array([[65],[66]],dtype='uint16'));x.attrs['MATLAB_class']=np.bytes_('char')
        cell=group.create_dataset('label',shape=(1,1),dtype=h5py.ref_dtype);cell[0,0]=x.ref
    result=read_mat(path)['data']
    assert result['eeg'].shape==(3,4);assert result['label']=='AB'

@pytest.fixture
def small_dataset(cfg,tmp_path):
    generate(tmp_path/'raw',subjects=4,trials=6,channels=8,muscles=3)
    cfg['data'].update(path=str(tmp_path/'raw'),format='npz',pattern='*.npz',units_confirmed=True,expected_eeg_channels=8,expected_emg_channels=3)
    cfg['split']['counts']=[2,1,1]
    cfg['paths'].update(runs=str(tmp_path/'runs'),models=str(tmp_path/'models'),processed=str(tmp_path/'processed.npz'))
    cfg['model']['random_forest'].update(n_estimators=8,max_depth=3,n_jobs=1)
    cfg['model']['mlp'].update(hidden=[12,6],max_epochs=8,patience=3)
    d=build_dataset(cfg);d.save(cfg['paths']['processed'])
    return Dataset.load(cfg['paths']['processed']),cfg

@pytest.mark.parametrize('kind',['ridge','random_forest','mlp'])
def test_end_to_end_and_reload(small_dataset,kind):
    from src.data.loader import iter_trials
    d,c=small_dataset;c['model']['name']=kind
    out,result,bundle=train(d,c,name=kind);loaded=load_model(out/'model')
    t=next(iter_trials(c));eeg,_=extract(t.eeg,2.,c['eeg']['window'],t.fs)
    epoch=EEGSample(eeg,t.fs,t.eeg_names,tuple(c['eeg']['window']))
    np.testing.assert_allclose(bundle.predict(epoch),loaded.predict(epoch),rtol=0,atol=0)
    np.testing.assert_allclose(bundle.features(epoch)[0],d.X[0])
    assert loaded.predict(epoch).shape==(3,)
    scaler=loaded.estimator.scaler if kind=='mlp' else loaded.estimator.named_steps['scaler']
    split=json.loads((out/'split.json').read_text())
    np.testing.assert_allclose(scaler.mean_,d.X[split['train']['indices']].mean(axis=0))
    assert result['synthetic'];assert (out/'plots/predicted_vs_true.png').exists()
    assert set(loaded.target_scaler.fitted_subjects_)==set(split['train']['subjects'])
    if kind=='mlp':
        scores=[r['validation_mse'] for r in loaded.estimator.history_]
        assert loaded.estimator.best_epoch_==np.argmin(scores)+1
    with pytest.raises(ValueError,match='channel order'):
        loaded.predict(EEGSample(eeg,t.fs,t.eeg_names[::-1],tuple(c['eeg']['window'])))
    changed=copy.deepcopy(c);changed['eeg']['bandpass'][0]=.5
    with pytest.raises(ValueError,match='mismatch'): train(d,changed,name='bad')

def test_mlp_gradient():
    from src.models.mlp import MLPRegressor
    for loss in ['mse','huber']:
        model=MLPRegressor({'loss':loss,'huber_delta':.5})
        pred=np.array([[.2,1.],[-2.,.3]]);y=np.zeros_like(pred)
        _,grad=model._loss_grad(pred,y)
        for idx in np.ndindex(pred.shape):
            a=pred.copy();b=pred.copy();a[idx]+=1e-6;b[idx]-=1e-6
            estimate=(model._loss_grad(a,y)[0]-model._loss_grad(b,y)[0])/2e-6
            assert grad[idx]==pytest.approx(estimate,abs=1e-7)


def test_future_eeg_with_derived_alignment(cfg,trial):
    cfg['alignment']['event']='emg_onset'
    a=sample(trial,cfg)
    other=copy.deepcopy(trial)
    other.eeg[:,a[2]['eeg_samples'][1]:]=1e5
    b=sample(other,cfg)
    np.testing.assert_array_equal(a[0],b[0])


def test_cancelled_run_status(small_dataset):
    d,c=small_dataset
    with pytest.raises(InterruptedError): train(d,c,name='cancelled',should_stop=lambda:True)
    status=json.loads((Path(c['paths']['runs'])/'cancelled/status.json').read_text())
    assert status['status']=='cancelled'
    assert not (Path(c['paths']['models'])/'cancelled').exists()


def test_huber_training(small_dataset):
    d,c=small_dataset;c['model']['name']='mlp';c['model']['mlp']['loss']='huber'
    _,_,model=train(d,c,name='huber')
    assert model.estimator.history_[-1]['train_loss'] < model.estimator.history_[0]['train_loss']


def test_outputs_cannot_enter_raw(cfg,tmp_path):
    from src.training.train import output_guard
    cfg['data']['path']=str(tmp_path/'raw')
    with pytest.raises(ValueError):output_guard(cfg,tmp_path/'raw'/'model')
