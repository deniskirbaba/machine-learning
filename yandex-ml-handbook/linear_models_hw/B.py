import numpy as np

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    assert y_true is not None, "ground truth vector should be non-empty"
    assert np.all(y_true >= 0), "ground thuth vector should have non-negative values (house prices cant be negative)"
    assert y_pred is not None, "prediction vector should be non-empty"
    assert len(y_true) == len(y_pred), "'y' vectors should be the same size"
    
    N = len(y_true)
    f = lambda x: max(a_min, x)
    y_pred = [f(i) for i in y_pred]
    res = np.sqrt(1 / N * np.sum(np.square(np.log(y_true) - np.log(y_pred))))
    return res