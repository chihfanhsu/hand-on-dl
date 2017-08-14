# 手把手的深度學習實務
## 預先下載
* 請安裝 [Anaconda2-4.3.1 或 Anaconda3-4.3.1](https://www.continuum.io/downloads)
* 預先下載[課程資料集 CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html)與[練習程式碼](https://drive.google.com/uc?id=0By9xOgk3An1wcFVCek9rSTIyQTA&export=download)

## Anaconda 安裝 (Windows 10, tensorflow)
* 安裝 anaconda3-4.3.1
* 使用"管理者權限"開啟 anaconda prompt
* （以下的輸入都在 anaconda prompt 下）
* 建立 python 3.5 的虛擬環境，並命名為 tensorflow，您的程式都需要在此環境下執行
```python
conda create -n tensorflow python=3.5
``` 
* 輸入 activate tensorflow 進入虛擬環境，您會看到命令列前出現 (tensorflow)
* （以下的輸入都在虛擬環境 tensorflow 下）
```python
activate tensorflow
```
* （二擇一）無 CUDA 支援的 GPU 請輸入
```python
pip install tensorflow
```
* （二擇一）有 CUDA 支援的 GPU 請輸入
```python
pip install tensorflow-gpu 
```
* 在 python 內利用此程式測試：
```python
import tensorflow as tf
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print (sess.run(c))
```
* 安裝需要的套件
```python
conda install numpy scipy mingw libpython
```
* 安裝 keras
```python
pip install keras
```
* 在 python 內測試 keras:
```python
import keras
``` 
* 請試跑 [CNN 測試程式](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py )

# GPU 安裝 (需要 NVIDIA 顯示卡)
## 在 Windows 10 安裝 CUDA8.0 & cuDNN5.1 可以參考下列網址
1. [安裝 CUDA&Theano](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/)
2. [安裝 cuDNN](http://ankivil.com/making-theano-faster-with-cudnn-and-cnmem-on-windows-10/)

## 在 ubuntu 上安裝可以參考下列影片
* https://www.youtube.com/watch?v=wjByPfSFkBo

# 沒有 GPU 的折衷方案 (Windows 10, OpenBLAS CPU 加速)
* 請參照 [windows_openblas.pdf](https://github.com/chihfanhsu/hand-on-dl/blob/master/windows_openblas.pdf) 進行安裝

## 課程額外需要的 libraries，請務必安裝
* pip (conda) install pillow (存圖用)
* pip (conda) install h5py (儲存 model)
* pip (conda) install future
* pip (conda) install matplotlib

## Python2 vs. Python3
* Python3 需將 print function ，添加上左右括號
```python
print()
```
* Python3 需將 unpickle function 改為
```python
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict
```

## 參考資料
* [Dataset (MNIST, CIFAR-10, CIFAR-100, STL-10, SVHN, ILSVRC2012 task 1) CNN 模型排行榜](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)

## Slides
* http://www.slideshare.net/tw_dsconf/ss-70083878
