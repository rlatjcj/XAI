import keras
import numpy as np

def saliency(c, model):
    _y = model.output[0, c]
    _grads = keras.backend.gradients(_y, model.input)[0]
    _saliency = keras.backend.function([model.input], [_grads])
    return _saliency

def smoothgrad(saliency, img_input, n=10, sigma=.01):
    _result = np.zeros_like(img_input)
    for i in range(n):
        _result += saliency([img_input + np.random.normal(scale=sigma, size=img_input.shape)])[0]
    return _result / n

def smoothgrad_square(saliency, img_input, n=10, sigma=.01):
    _result = np.zeros_like(img_input)
    for i in range(n):
        _result += saliency([img_input + np.random.normal(scale=sigma, size=img_input.shape)])[0]**2
    return _result / n

def vargrad(saliency, img_input, n=10, sigma=.01):
    _smoothgrad = smoothgrad(saliency, img_input, n, sigma)
    _result = np.zeros_like(img_input)
    for i in range(n):
        _result += saliency([img_input + np.random.normal(scale=sigma, size=img_input.shape)])[0] - _smoothgrad**2
    return _result / n

def integratedgrad(saliency, img_input, step=50, baseline=None):
    _input = img_input[np.newaxis,...]/255
    
    if baseline is None:
        baseline = 0 * img_input
        
    assert baseline.shape == img_input.shape
    
    _inputs = [baseline+(float(j)/step)*(img_input-baseline) for j in range(step+1)]
    _temp = np.zeros_like(img_input)
    for inp in _inputs:
        _temp += saliency([inp])[0]
    return (img_input - baseline) * _temp / step