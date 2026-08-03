# MSAdapter


简体中文

MSAdapter是一款MindSpore生态适配工具，在不改变用户原有使用习惯下，将PyTorch/JAX等三方框架代码快速迁移到MindSpore生态上，帮助用户高效使用昇腾算力。

### 简介

msadapter是将PyTorch训练脚本高效迁移至MindSpore框架执行的工具，其目的是在不改变原有PyTorch用户的使用习惯情况下，使得PyTorch代码能在昇腾上获得高效性能。


- **PyTorch接口支持**： msadapter目前支持大部分PyTorch常用接口适配。用户接口使用方式不变，基于MindSpore动态图或静态图模式下执行在昇腾算力平台上。可以在[torch接口支持列表](#TODO)中查看接口支持情况。


### 文档
有关安装指南、教程和API的更多详细信息，请参阅[教程文档](#TODO)。

## msadapter入门指南

### 安装
首先查看[版本说明](#版本说明)选择所需的msadapter和MindSpore版本。

#### 安装MindSpore
请根据MindSpore官网[安装指南](https://www.mindspore.cn/install)进行安装。


#### 安装msadapter

##### 方式一：使用源码, 环境变量切换

```bash
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/msa_thirdparty:$PYTHONPATH
```

##### 方式二：安装包

步骤一：源码下载
```bash
 git clone https://gitee.com/mindspore/msadapter.git
```
步骤二：构建
```bash
 cd msadapter
 bash scripts/build.sh
```
构建完成后，msadapter目录下会新增一个build文件夹与一个dist文件夹。

步骤三：安装

```bash
pip install ${MindSpeed_Core_MS_PATH}/msadapter/dist/*.whl
export PYTHONPATH=/*/site-packages/msa_thirdparty:$PYTHONPATH 
# /*/site-packages 指python环境下的安装包路径，可以使用pip show msadapter获取。
```
步骤四：脚本前加一行，切换后端
```python
import msadapter # 改为mindspore后端执行
import torch
from torch.nn import functional as F

```

脚本中控制是否使用msadapter的代码
```python
msadapter.enable_torch_proxy(True)
msadapter.enable_torch_proxy(False)
```

### 使用msadapter

#### 通过环境变量使能msadapter，脚本使用PyTorch
```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Linear(10, 1)
```

#### 脚本使用PyTorch，最开始的位置引用msadapter
```python
import msadapter # 改为mindspore后端执行

import torch
from torch import nn
from torch.nn import functional as F

net = nn.Linear(10, 1)
```

#### 脚本直接使用msadapter
```python
import msadapter
from msadapter import nn
from msadapter.nn import functional as F

net = nn.Linear(10, 1)
```

### 多机器场景下启动方式：
**需要使用msrun启动，不支持使用torchrun启动。**
* msrun和torchrun入参不同, 参数样例：
```bash
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6099
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --log_dir=msrun_log \  # 分卡日志路径存储路径
    --join=True \  # True代表显示屏打日志
"

msrun $DISTRIBUTED_ARGS pretrain_gpt.py
```




安装好msadapter后, 你可以按照以下方式使用它:

  <details>
    <summary><b>直接import torch即可将PyTorch代码适配到msadapter,在NPU设备上运行PyTorch代码</b></summary>

   ``` python
    import msadapter
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    # 1.Working with data
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    # 2.Creating Models
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    if __name__ == '__main__':
        train_dataloader = DataLoader(training_data, batch_size=64)
        test_dataloader = DataLoader(test_data, batch_size=64)

        # 3.create Models
        model = NeuralNetwork()

        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        # 4.Predict
        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
   ```

   </details>

\
安装完msadapter后，代码执行时torch同名的导入模块会自动被转换为msadapter相应的模块（目前支持torch、torchvision、torch_npu、torchair等相关模块的自动转换），接下来执行主入口的.py文件即可。更多的使用方式可以参考[使用指南](#TODO)

### 版本说明

| **分支名** | **发布版本**  | **发布时间**          | **配套MindSpore版本**        |
|--------------|----------------|--------------------|-------------------------|
| **master** | -    | -           |  [MindSpore master](https://www.mindspore.cn/install) |


- MindSpore版本推荐从[MindSpore官网](https://www.mindspore.cn/versions)获取。


## 限制
目前MSAdapter的使用存在如下限制：

  <details>
    <summary>暂不支持Complex64/Complex128</summary>

   示例代码：
   ``` python
     from torch.utils.data import DataLoader
     from torchvision import datasets
     from torchvision.transforms import ToTensor

     training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
     train_dataloader = DataLoader(training_data, batch_size=64, pin_memory=True)
     for batch, (X, y) in enumerate(train_dataloader):
         X, y = X.cuda(), y.cuda()
   ```
   报错信息如下：
   ``` python
    Traceback (most recent call last):
        File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 98, in pin_memory
            clone[i] = pin_memory(item, device)
        File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 64, in pin_memory
            return data.pin_memory(device)
    TypeError: pin_memory() takes 1 positional argument but 2 were given
   ```

  </details>

  <details>
    <summary>Dataloader中的pin_memory参数仅支持设置为False</summary>

   示例代码：
   ``` python
     from torch.utils.data import DataLoader
     from torchvision import datasets
     from torchvision.transforms import ToTensor

     training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
     train_dataloader = DataLoader(training_data, batch_size=64, pin_memory=True)
     for batch, (X, y) in enumerate(train_dataloader):
         X, y = X.cuda(), y.cuda()
   ```
   报错信息如下：
   ``` python
    Traceback (most recent call last):
        File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 98, in pin_memory
            clone[i] = pin_memory(item, device)
        File "/path/to/your/torch/utils/data/_utils/pin_memory.py", line 64, in pin_memory
            return data.pin_memory(device)
    TypeError: pin_memory() takes 1 positional argument but 2 were given
   ```
  </details>

  <details>
    <summary>MindSpore导出的ckpt文件无法被直接加载到PyTorch模型中</summary>

   示例代码：
   ``` python
    import torch
    from torch import nn
    import mindspore as ms

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(28*28, 512)

        def forward(self, x):
            logits = self.linear(x)
            return logits

    class myNN(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(28*28, 512)

        def construct(self, x):
            logits = self.linear(x)
            return logits

    model = myNN()
    ms.save_checkpoint(model, "./net.ckpt")
    model2 = NeuralNetwork()
    model.load_state_dict(torch.load("./net.ckpt"))
   ```
   报错信息如下：
   ``` python
    Traceback (most recent call last):
        File "/path/to/your/demo.py", line 99, in <module>
            model.load_state_dict(torch.load("./mynn.ckpt"))
        File "/path/to/your/torch/serialization.py", line 1020, in load
            return _legacy_load(opened_file, pickle_module, **pickle_load_args)
        File "/path/to/your/torch/serialization.py", line 1118, in _legacy_load
            magic_number = pickle_module.load(f, **pickle_load_args)
    EOFError: Ran out of input
   ```

  </details>

  <details>
    <summary>不支持MindSpore与MS-Adapter混合运行</summary>
   import torch后，mindspore的部分行为会变更为torch的行为，从而产生不可预期的错误。

   示例代码：
   ``` python
    from mindspore import Tensor

    a = Tensor([2, 2])
    print(f'before import torch: a.shape={a.shape}')

    import torch
    print(f'after import torch: a.shape={a.shape}')

   ```
   执行结果如下，可以看到，import torch后，原本的mindspore.Tensor.shape行为发生了改变。
   ``` python
    before import torch: a.shape=(2,)
    after import torch: a.shape=torch.Size([2])
   ```
   不支持混跑的MindSpore接口详见下表:
<table>
    <tr>
        <th><b>模块</b></th>
        <th><b>受影响接口</b></th>
    </tr>
    <tr>
        <td rowspan="55">mindspore.Tensor/mindspore.StubTensor</td>
        <td>is_shared</td>
    </tr>
    <tr>
        <td>softmax</td>
    </tr>
    <tr>
        <td>type_</td>
    </tr>
    <tr>
        <td>retain_grad</td>
    </tr>
    <tr>
        <td>shape</td>
    </tr>
    <tr>
        <td>to_dense</td>
    </tr>
    <tr>
        <td>_base</td>
    </tr>
    <tr>
        <td>data</td>
    </tr>
    <tr>
        <td>numel</td>
    </tr>
    <tr>
        <td>nelement</td>
    </tr>
    <tr>
        <td>repeat</td>
    </tr>
    <tr>
        <td>cuda</td>
    </tr>
    <tr>
        <td>npu</td>
    </tr>
    <tr>
        <td>cpu</td>
    </tr>
    <tr>
        <td>size</td>
    </tr>
    <tr>
        <td>dim</td>
    </tr>
    <tr>
        <td>clone</td>
    </tr>
    <tr>
        <td>log_softmax</td>
    </tr>
    <tr>
        <td>narrow</td>
    </tr>
    <tr>
        <td>view</td>
    </tr>
    <tr>
        <td>__or__</td>
    </tr>
    <tr>
        <td>device</td>
    </tr>
    <tr>
        <td>__and__</td>
    </tr>
    <tr>
        <td>__xor__</td>
    </tr>
    <tr>
        <td>__iter__</td>
    </tr>
    <tr>
        <td>__reduce_ex__</td>
    </tr>
    <tr>
        <td>expand</td>
    </tr>
    <tr>
        <td>detach</td>
    </tr>
    <tr>
        <td>T</td>
    </tr>
    <tr>
        <td>transpose</td>
    </tr>
    <tr>
        <td>mean</td>
    </tr>
    <tr>
        <td>clamp</td>
    </tr>
    <tr>
        <td>is_cuda</td>
    </tr>
    <tr>
        <td>is_cpu</td>
    </tr>
    <tr>
        <td>repeat_interleave</td>
    </tr>
    <tr>
        <td>is_sparse</td>
    </tr>
    <tr>
        <td>requires_grad</td>
    </tr>
    <tr>
        <td>requires_grad_</td>
    </tr>
    <tr>
        <td>unsqueeze</td>
    </tr>
    <tr>
        <td>__pow__</td>
    </tr>
    <tr>
        <td>float</td>
    </tr>
    <tr>
        <td>backward</td>
    </tr>
    <tr>
        <td>expand</td>
    </tr>
    <tr>
        <td>split</td>
    </tr>
    <tr>
        <td>norm</td>
    </tr>
    <tr>
        <td>record_stream</td>
    </tr>
    <tr>
        <td>data_ptr</td>
    </tr>
    <tr>
        <td>pin_memory</td>
    </tr>
    <tr>
        <td>grad</td>
    </tr>
    <tr>
        <td>grad</td>
    </tr>
    <tr>
        <td>__imul__</td>
    </tr>
    <tr>
        <td>reshape</td>
    </tr>
    <tr>
        <td>squeeze</td>
    </tr>
    <tr>
        <td>element_size</td>
    </tr>
    <tr>
        <td>exponential_</td>
    </tr>
</table>

  </details>


## 许可证
[Apache License 2.0](https://openi.pcl.ac.cn/OpenI/MSAdapter/src/branch/master/LICENSE)