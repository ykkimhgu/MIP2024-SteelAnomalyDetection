# MIP2024-Unsupervised Anomaly Detection for Steel Surface Inspection

# 1. Introduction

본 repository는 2024년 1학기에 진행된 기전융합종합 설계, **Comparison Study of Unsupervised Deep Learning Techniques for Steel Surface Inspection**주제의 연구에 대한 software 소스 파일과 환경 구축 내용으로 구성되어 있습니다.

연구의 목적: 철강의 표면 이상치 탐지를 위한 최신 딥러닝 모델에 대한 Survey를 진행한 뒤, 표준 데이터셋인 MVTec 데이터셋에 적용해보고 해당 모델을 실제 철강 데이터셋인 Severstal 데이터에 적용해보며 성능을 평가해보는 것을 목표로 합니다.



# 2. 환경 구축

본 Repository에서는 크게 세 가지 딥러닝 모델 (**Convolutional autoencoder, PatchCore, DRAEM**)을 다룹니다.

세 모델 모두 적용하는데 있어, GPU 자원을 활용하는 것이 효율적이기 때문에, CUDA, cuDNN, pyTorch 설치가 필요합니다.

* PC GPU: Geoforce RTX 3070

* CUDA: Version **11.8**
* cuDNN: Version **8.6.0** for CUDA 11.x
* pyTorch: Version **2.1.2**



## 2.1. CUDA 설치

* CUDA version: 11.8

* 설치 가능한 최대 CUDA 버전 확인

  1. Windows 창에 `cmd`검색

  2. cmd 창에 `nvidia-smi`명령어 실행

     <img src="https://github.com/HanMinung/CAE/assets/99113269/b52641be-909f-4c7d-820c-d6ba512028ec" alt="image" style="zoom: 67%;" />

  3. 우측 상단의 CUDA Version이 호환 가능한 최대 CUDA version 입니다.

* 설치 link: https://developer.nvidia.com/cuda-toolkit-archive

  

## 2.2. cuDNN 설치

* cuDNN version: v8.6.0 for CUDA 11.x

* 설치 링크: https://developer.nvidia.com/rdp/cudnn-archive

* 설치하게 되면, cuDNN 폴더 내 bin, include, lib 파일을 복사하여 그대로 CUDA 설치 경로에 복사

* 작성자의 CUDA 설치 경로는 다음과 같습니다.

  **C drive - Program Files - NVIDIA GPU Computing Toolkit - CUDA - v11.8**

* 위 과정이 정상적으로 이루어지게 되면, cmd 창에 `nvcc -V`명령어를 입력하게 되면 다음과 같이 결과가 나오게 됩니다.

  <img src="https://github.com/HanMinung/CAE/assets/99113269/e8e3ac91-9358-4763-8fc7-9b062e1b1941" alt="image" style="zoom: 80%;" />



## 2.3. PyTorch 설치

* **2.1, 2.2** 과정을 마쳤다면, 마지막으로 pytorch를 설치해야 합니다. 

* pytorch에는 CPU 버전과, GPU 버전이 있는데, GPU 활성화를 위해 GPU 버전의 설치를 진행합니다.

* 먼저, anaconda에서 구축한 본인의 가상환경을 실행

  ```python
  conda activate [environment name]
  ```

* 후에, 다음 명령어를 conda에서 실행함으로써, pytorch를 설치할 수 있습니다.

  ```python
  conda install -c anaconda cudatoolkit=11.8 cudnn seaborn jupyter
  conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install torchsummary
  ```

* 정상적으로 설치가 됐다면, 가상환경이 활성화 된 상태에서, 다음 명령어를 입력합니다.

  ```python
  conda activate [environment name] 		# 가상환경 활성화
  python
  import torch
  torch.cuda.is_available()
  ```

* 위 명령어를 실행했을때, `True`라는 결과가 나오게 되면, 모두 정상적으로 환경 구축이 된 것입니다.

  <img src="https://github.com/HanMinung/CAE/assets/99113269/835f5828-a8ce-4cf3-973e-2432656dd5f0" alt="image" style="zoom:80%;" />



## 2.4. 나머지 라이브러리 설치

위 과정을 통해, 코드를 구동하는데 있어 GPU 활용이 가능하며 Convolutional autoencoder, PatchCore, DRAEM 모델 마다 추가적으로 설치가 필요한 라이브러리에 대해 다루고자 합니다. 2.3의 Pytorch를 제외한 대부분의 라이브러리는 파이썬 터미널 내 `PIP install` 명령어를 활용하여 설치를 진행했습니다.

같은 가상환경 내에서 라이브러리를 구축한 뒤, 세 코드 모두 같은 interpreter를 활용했기 때문에, 공통되는 라이브러리는 제외하도록 하겠습니다.

* Convolutional autoencoder 라이브러리 목록

  ```python
  pip install matplotlib 				# version 3.8.4
  pip install os						
  pip install numpy					# version 1.26.4
  pip install opencv					# version 4.9.0
  ```

  

* PatchCore 라이브러리 목록

  ```python
  pip install common
  pip install backbones				# To use pretrained encoder
  pip install tqdm					# version 4.66.2
  pip install PIL						
  pip install faiss-cpu				# GPU 버전이 따로 설치되지 않아, CPU 버전을 설치 
  pip install scipy
  ```



* DRAEM 라이브러리 목록

  ```python
  pip install imgaug					# version 0.4.0
  pip install math					# version 1.3.0
  pip install sklearn					# version 1.4.1
  ```

  

위 라이브러리가 모두 정상적으로 설치한 뒤, 확인을 위해 anaconda 상에서 다음 명령어를 입력함으로써 정상 설치 여부를 검증할 수 있습니다.

```python
conda activate [environment name]
conda list
```





## 3. Dataset 설치 및 참고 Github

### Dataset 설치

----

먼저, anomaly detection 파트에 있어 논문에서 모델 검증을 위해 가장 많이 활용되는 데이터셋은 **MVTec** 데이터셋입니다.

참고한 논문에서 모두 모델의 검증을 위해 해당 데이터셋을 활용했습니다.

* MVTec 데이터셋 설치 경로: https://www.mvtec.com/company/research/datasets/mvtec-ad




또한, DRAEM 모델의 경우, anomaly source 이미지를 생성하기 위해, **dtd (discribable textures data)**이미지라는 것을 활용하는데, 해당 데이터셋의 설치 경로는 다음과 같습니다.

* dtd 데이터셋 설치 경로: https://www.robots.ox.ac.uk/~vgg/data/dtd/



마지막으로, 최종적으로 적용하는 Severstal dataset의 설치 경로는 아래와 같습니다.

* Severstal 데이터셋 설치 경로: https://www.kaggle.com/c/severstal-steel-defect-detection



### 참고 Github

---

먼저, Convolutional autoencoder의 경우, 직접 데이터셋의 형태에 맞게 코드를 구축하는 것이 맞기 때문에, 아래 Blog 요소들을 참고하여 직접 구축했습니다.

#### 1) Convolutional autoencoder

* 참고 블로그 1 (코드 구축): https://zir2-nam.tistory.com/entry/025-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94-%EC%95%84%EC%8B%B8-%EC%A2%8B%EA%B5%AC%EB%82%98
* 참고 블로그 2 (Anomaly detection 방식): https://hoya012.github.io/blog/anomaly-detection-overview-1/



다음으로, PatchCore 모델의 경우 official github source가 없었기 때문에, 고려대학교 DSBA 연구실의 github source를 참고했습니다.

#### 2) Patchcore

* 참고 Github: https://github.com/hwk0702/keras2torch/tree/main/Anomaly_Detection/PatchCore



마지막으로, DRAEM 모델의 경우, official github source를 활용했으며, 기존에는 Ubuntu 환경에서 코드를 실행하나 저는 Windows 환경에서 코드를 돌리기 위해 .sh 확장자의 실행 파일들은 따로 사용하지 않았습니다.

#### 3) DRAEM

* Official Github: https://github.com/vitjanz/draem



## 4. 각 코드 Description

각 코드에 대한 모든 내용을 다루기는 어렵고, 제가 따로 구축하거나 각 코드의 핵심적인 부분에 대한 내용은 다음과 같습니다.



### 4.1. Convolutional autoencoder

* Encoder와 Decoder의 구조 (class ConvAutoencoder)

  아래 코드는 MVTec 데이터를 (256, 256) 사이즈로 resize하여 활용하는 형태에 맞춰진 것이며, 후에 다른 데이터를 활용하는데 있어서는 구조의 수정이 필요합니다. 코드의 encoder, decoder 구조는 아래와 같이 설계되었습니다.

  <img src="https://github.com/HanMinung/CAE/assets/99113269/28708d99-9f24-45a8-9771-b686ed60369e" alt="image" style="zoom:67%;" />

  ```python
  class ConvAutoencoder(nn.Module):
      def __init__(self):
          super(ConvAutoencoder, self).__init__()
          # Encoder
          self.encoder = nn.Sequential(
              nn.Conv2d(3, 64, 3, padding=1),   # [B, 64, 256, 256]
              nn.ReLU(),
              nn.MaxPool2d(2, 2, padding=0),    # [B, 64, 128, 128]
              nn.Conv2d(64, 128, 3, padding=1), # [B, 128, 128, 128]
              nn.ReLU(),
              nn.MaxPool2d(2, 2, padding=0),    # [B, 128, 64, 64]
              nn.Conv2d(128, 256, 3, padding=1),# [B, 256, 64, 64]
              nn.ReLU(),
              nn.MaxPool2d(2, 2, padding=0)     # [B, 256, 32, 32]
          )
          # Decoder
          self.decoder = nn.Sequential(
              nn.Conv2d(256, 256, 3, padding=1),# [B, 256, 32, 32]
              nn.ReLU(),
              nn.Upsample(scale_factor=2),      # [B, 256, 64, 64]
              nn.Conv2d(256, 128, 3, padding=1),# [B, 128, 64, 64]
              nn.ReLU(),
              nn.Upsample(scale_factor=2),      # [B, 128, 128, 128]
              nn.Conv2d(128, 64, 3, padding=1), # [B, 64, 128, 128]
              nn.ReLU(),
              nn.Upsample(scale_factor=2),      # [B, 64, 256, 256]
              nn.Conv2d(64, 3, 3, padding=1),   # [B, 3, 256, 256]
              nn.Sigmoid()
          )
      
      def forward(self, x):
          x = self.encoder(x)
          x = self.decoder(x)
          return x
  ```



## 4.2. DRAEM 모델

* 학습 조건 설정 (Parameter 설정 / 파이썬 파일을 실행할 때, 따로 인자로 받는 것이 아니라 default로 설정해 줬습니다!)

  * bs: batch size (원본 코드에서는 8로 설정, 컴퓨터 스펙에 맞게 설정)
  * lr: learning rate
  * epoch: 학습 수
  * data_path: MVTec 데이터셋 저장 경로 (상대 경로 활용)
  * anomaly_source_pathsource path: **dtd (discriminable textures datasets)** class 중 anomaly source로 활용한 데이터셋
  * checkpoint_path: 학습된 모델 저장 경로 (**.pckl, _seg.pckl 확장자의 두 파일이 자동 저장**)

  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument('--obj_id', action='store', type=int, required=False, default=1)
  parser.add_argument('--bs', action='store', type=int, required=False, default=2)
  parser.add_argument('--lr', action='store', type=float, required=False, default=0.0001)
  parser.add_argument('--epochs', action='store', type=int, required=False, default=700)
  parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
  parser.add_argument('--data_path', action='store', type=str, required=False, default="../Datasets\\MVTec\\")
  parser.add_argument('--anomaly_source_path', action='store', type=str, required=False, default="../Datasets\\dtd\\images\\banded\\")
  parser.add_argument('--checkpoint_path', action='store', type=str, required=False, default="checkpoints\\")
  parser.add_argument('--log_path', action='store', type=str, required=False, default="logs\\")
  parser.add_argument('--visualize', action='store_true')
  args = parser.parse_args()
  ```



* 설정할 hyperparameter 종류 
  * batch size: **batch size 8을 기준**으로 했을 때, 학습하는 중에 GPU는 일반적으로 **10GB ~ 12GB 정도 실시간으로 소모**되었기에, 학습 컴퓨터 성능에 맞게 설정하는 과정이 필요합니다.
  * Learning rate: 모델 내부가 U-net 구조를 가지고 있기 때문에, 0.0001을 설정하는 것이 적절하다고 판단했으나, 후에 조정이 필요하다면 수정이 필요한 파라미터입니다.
  * SSIM loss에서의 window size: 이미지의 window size에 맞게 커팅하고, 해당 영역이 얼마나 비슷한지를 판단하게 되는데, 원본 코드에서는 해당 수치를 11로 설정.























