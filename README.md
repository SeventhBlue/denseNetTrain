# denseNetTrain
ocr中的densenet网络训练
## 简介
原项目是：https://github.com/YCG09/chinese_ocr   这个只是提供原项目中的densenet网络训练用的。原项目中没有fine-tune，
同时tensorboard使用时好像有点问题（不过使用tensorboard好像也看不到有用的信息）

## 环境部署
``` Bash
sh setup.sh
```
* 注：支持window和Ubuntu

#### 1. 数据准备

数据集：https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS1Pw (密码：lu7m)
* 共约364万张图片，按照99:1划分成训练集和验证集
* 数据利用中文语料库（新闻 + 文言文），通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成
* 包含汉字、英文字母、数字和标点共5990个字符
* 每个样本固定10个字符，字符随机截取自语料库中的句子
* 图片分辨率统一为280x32

图片解压到相应的位置，注意路径

#### 2. 训练
将train.py和fine_tune_train.py中路径改成自己准备好的数据的位置
