# 手把手的深度學習實務
## 預先下載
* 請安裝 [Anaconda 2.7](https://www.continuum.io/downloads)
* 請依據[預先下載](https://github.com/chihfanhsu/dnn_hand_by_hand/blob/master/cnn_preDL.pdf)事先下載課程資料

# GPU 安裝 (需要 NVIDIA 顯示卡)
## 在 Windows 10 安裝 CUDA & cuDNN 可以參考下列網址
1. [安裝 CUDA&Theano](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/)
2. [安裝 cuDNN](http://ankivil.com/making-theano-faster-with-cudnn-and-cnmem-on-windows-10/)

## 在 ubuntu 上安裝可以參考下列影片，建議安裝 CUDA 7.5
* https://www.youtube.com/watch?v=wjByPfSFkBo

# 沒有 GPU 的折衷方案 (Windows 10, OpenBLAS CPU 加速)
* 請參照[預先下載](https://github.com/chihfanhsu/hand-on-dl/blob/master/cnn_preDL.pdf)的第四步驟進行安裝

## 使用 library
* pip(3) install pillow
* pip(3) install scipy
* pip(3) install numpy
* pip(3) install h5py

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
