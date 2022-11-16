## 前言

&emsp;&emsp;首先感谢bubbliiiing博主提供的训练代码！
整个部署程序分为四步（C++）：
1. 下载bubbliiiing博主项目更改predict 为 export_onnx
2. 更改文件夹nets下的yolo.py代码
3. 生成onnx文件
4. 下载此项目进行cmake&make
经测试，可以对(pytorch版本的)yolov4 -- yolov5 -- yolov5-6.1 yolov7 进行转换测试
由于bubbliiiing博主已经将网络结构和如何训练，我这边就不再过多的讲述如何训练。

## 如何使用？
&emsp;&emsp;这里我以yolov7为例子向大家展示如何得到onnx以及运行onnx进行部署(bubbliiiing博主v4>>v5>>v5-6.1同理)

## Step1:
git clone https://github.com/bubbliiiing/yolov7-pytorch.git
cd yolov7-pytorch\
pip install -r requirements.txt\
vim predict.py
```
line24: mode = "export_onnx" ## predict >> 'export_onnx'
```
## Step2:
cd /nets\
vim yolo.py
```
line300 ----line316:
        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]
```
如下图为bubbliiiing博主源码生成的onnx输入输出信息截取

![image](https://user-images.githubusercontent.com/59249258/202062401-2d119b52-c7ac-4956-b333-2fde7b5b205e.png)


&emsp;&emsp;bubbliiiing 这里的return[out0, out1, out2] 在经过"export_onnx"后并没有torch.cat，我们需要将输出的结果进行torch.cat操作，
在进行torch.cat操作之前需要将输出结果进行转换一下维度，详见如下代码：
```
        out2 = self.yolo_head_P3(P3)
        bs, _, ny, nx = out2.shape  
        
        # 这里的 no = 85[类别数nc（80） + 目标坐标和置信度（4+1）]
        
        out2 = out2.view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # no = 85
        out2 = out2.view(bs * 3 * ny * nx, 85, 1).contiguous() #  no = 85
        # ---------------------------------------------------#
        #   第二个特征层
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        bs, _, ny, nx = out1.shape
        out1 = out1.view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #  no = 85
        out1 = out1.view(bs * 3 * ny * nx, 85, 1).contiguous() #  no = 85
        # ---------------------------------------------------#
        #   第一个特征层
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        bs, _, ny, nx = out0.shape
        out0 = out0.view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #  no = 85
        out0 = out0.view(bs * 3 * ny * nx, 85, 1).contiguous() #  no = 85

        output = torch.cat((out2, out1, out0))
        output = output.permute(2, 0, 1)

        # return [out0, out1, out2]
        return output
```
&emsp;&emsp;**使用上述的代码部分替代源代码的line300 至 line316:** 更换后得到的信息下图所示\
![image](https://user-images.githubusercontent.com/59249258/202062358-0f725441-1633-46da-996d-15d689f04a67.png)


&emsp;&emsp;大家在更换自己的数据集进行训练得到的权重进行pth2onnx时需要注意：上述代码注释部分的更换85（coco数据集的类别数+5），
在这里需要更换为自己的数据集的类别数再加上5。例如自己的数据集是10个类别(nc=10)，那么这里的no=10+5 ;也就是85需要更改为15（no = nc + 5）

## Step3:
cd ..\
python predict.py \
这样我们就可以得到C++调用onnx需要的权重

## step4: 
git clone https://github.com/kivenyangming/yolo-bubbliiiing-onnx.git \
cd yolo-bubbliiiing-onnx\
mkdir build\
cd build\
cmake ..\
cmake --build .    (或者:make)
