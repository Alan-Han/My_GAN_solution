## 1.	运行环境 

ios 10.13  python 3.5.4  tensorflow 1.2.0


## 2. 获取数据集并预处理：

(1)运行主程序main.py后，调用input_dataset.py自动下载数据集，数据集包含600000张32x32x3的街景门牌号照片及对应标签，下载地址为http://ufldl.stanford.edu/housenumbers/train_32x32.mat

(2)数据集下载后运行gan.py调用GAN类建立网络，并在主程序中调用Dataset类对数据进行归一化预处理，图像归一化至0~1


## 3. 训练并测试模型

(1)模型存放在gan.py，generator采用4个反卷积层，前3层用transposed convolution > batch norm > leaky ReLU的架构，最后1层接tanh激活输出，discriminator采用4个卷积层，中间2层用convolution > batch norm > leaky ReLU的架构，最后1层接sigmoid激活输出

(2)主程序中运行train()开始训练并测试模型，设置AdamOptimizer作为优化器，训练25次，最终测试集准确率可达到70%

