# 实时疲劳检测系统

基于传统计算机视觉的轻量级实时疲劳检测系统，使用人脸关键点和人工设计特征，无需深度学习模型。

## 目录
- [实时疲劳检测系统](#实时疲劳检测系统)
  - [目录](#目录)
  - [安装依赖](#安装依赖)
  - [使用方法](#使用方法)
  - [项目结构](#项目结构)
  - [检测指标](#检测指标)
  - [备注](#备注)




## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 实时摄像头检测
python monitor.py

# 图片测试
python image_test.py
```

## 项目结构
```
fatigue-detection/
├── face_analyzer.py         # 面部关键点检测
├── fatigue_detector.py      # 疲劳分析
├── monitor.py               # 主程序，调用摄像头进行实时检测
├── image_test.py            # 图片测试
├── notebooks/               # Jupyter版本代码
├── models/                  # 保存人脸关键点检测模型
├── requirements.txt
└── README.md
```

## 检测指标
- 眼睛闭合程度
- 打哈欠检测  
- 头部姿态
- 头部运动

按 'q' 退出程序，按 'r' 重置统计。

## 备注
主要是检测闭眼和打哈欠。阈值可以自己调整，代码中采用的并非最优参数。一时兴起让DeepSeek写的，很简单且不完善，是作为课程设计都磕碜的程度，干脆不优化了。作用大概是给学CS的萌新增加信心？