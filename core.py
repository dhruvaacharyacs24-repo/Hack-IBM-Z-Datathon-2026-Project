import os, time
import numpy as np, torch, torch.nn as nn
from scipy.signal import butter, sosfiltfilt

FS, WIN = 360, 1080
LEVELS = ['Normal', 'Watch', 'Critical']
CRIT_T = 0.9   # Critical only if P(Critical) > CRIT_T
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecg_cnn_v2.pt')   # swap to ecg_cnn.pt for v1
_sos = butter(3, [0.5, 40], btype='band', fs=FS, output='sos')

def preprocess(w):
    w = sosfiltfilt(_sos, np.asarray(w, dtype=np.float64))
    return ((w - w.mean()) / (w.std() + 1e-6)).astype(np.float32)

class ECGNet(nn.Module):        # same weights as ECGNet2 (dropout has no parameters and is off in eval)
    def __init__(s):
        super().__init__()
        blk = lambda i, o, k: nn.Sequential(nn.Conv1d(i, o, k, padding=k//2), nn.BatchNorm1d(o), nn.ReLU())
        s.f = nn.Sequential(blk(1,32,7), nn.MaxPool1d(2), blk(32,64,5), nn.MaxPool1d(2),
                            blk(64,128,5), nn.MaxPool1d(2), blk(128,128,3))
        s.h = nn.Linear(128, 3)
    def forward(s, x): return s.h(s.f(x).mean(-1))

_model = None
def get_model(path=CKPT):
    global _model
    if _model is None:
        m = ECGNet(); m.load_state_dict(torch.load(path, map_location='cpu')); m.eval(); _model = m
    return _model

def to_tensor(raw_win):
    assert len(raw_win) == WIN, f'expected {WIN} samples, got {len(raw_win)}'
    return torch.tensor(preprocess(raw_win))[None, None]

def predict(raw_win, model=None):
    t0 = time.perf_counter()
    if model is None: model = get_model()
    with torch.no_grad():
        probs = torch.softmax(model(to_tensor(raw_win)), 1)[0].numpy()
    q = probs.copy()
    if q[2] <= CRIT_T: q[2] = 0
    return {'level': LEVELS[int(q.argmax())], 'probs': probs,
            'risk': float(1 - probs[0]), 'latency_ms': (time.perf_counter() - t0) * 1000}
