# Chainer Examples

### install cupy (for GPU)

.bashrc
```shell
export PATH="/usr/local/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export CUDA_PATH=/usr/local/cuda-8.0
export CFLAGS=-I/usr/local/cuda-8.0/include
export LDFLAGS=-L/usr/local/cuda-8.0/lib64;
```

Then,
```shell
source ~/.bashrc
pip install cupy
```

## useful libraries
<a href='https://github.com/cupy/cupy'>CuPy</a>
<a href='https://github.com/mdbloice/Augmentor'>Augmentor</a>
<a href='https://github.com/neka-nat/tensorboard-chainer.git'>tensorboard-chainer</a>

If you use tensorboard-chainer,
```shell
pip install tensorflow
pip install tensorboard
pip install git+https://github.com/neka-nat/tensorboard-chainer
```

## please look at <a href='https://github.com/fytroo/Chainer-Examples/blob/master/flower_lenet_tb.ipynb'>flower_lenet_tb.ipynb</a>

 
