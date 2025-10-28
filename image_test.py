import os
import cv2
import matplotlib.pyplot as plt
from fatigue_detector import FaceAnalyzer

def visualize_features():
    """可视化特征计算过程"""
    # 加载本地人脸图片
    image_path = "Test_image_from_NTHU_DDD.jpg"  # 修改为您的本地图片路径
    
    if not os.path.exists(image_path):
        print(f"⚠️  未找到测试图片: {image_path}")
        print("请将图片路径修改为您的本地人脸图片路径")
        return
    
    # 读取图片
    test_img = cv2.imread(image_path)
    if test_img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    # 初始化面部分析器
    landmark_model_path = "./models/shape_predictor_68_face_landmarks.dat"  # 修改为您的模型路径
    analyzer = FaceAnalyzer(landmark_model_path)
    
    # 检测人脸和关键点
    face, landmarks = analyzer.detect_face_and_landmarks(test_img)
    
    if landmarks is None:
        print("❌ 未检测到人脸关键点")
        # 显示原图
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image - No Face Detected")
        plt.axis('off')
        plt.show()
        return
    
    # 绘制关键点
    result_img = analyzer.draw_landmarks(test_img.copy(), landmarks, color=(0, 255, 0))
    
    # 提取特征
    features = analyzer.extract_fatigue_features(landmarks)
    
    # 在图像上添加特征信息
    y_offset = 30
    feature_texts = [
        f"Eye Closure: {features['eye_closure']:.3f}",
        f"Mouth Openness: {features['mouth_openness']:.3f}",
        f"Head Pose: {features['head_pose']:.1f} deg"
    ]
    
    for text in feature_texts:
        cv2.putText(result_img, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # 显示结果
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Facial Landmarks and Fatigue Features")
    plt.axis('off')
    plt.show()
    
    # 打印详细特征信息
    print("=== 面部疲劳特征分析 ===")
    print(f"眼睛闭合程度: {features['eye_closure']:.3f}")
    print(f"嘴部张开程度: {features['mouth_openness']:.3f}")
    print(f"头部姿态角度: {features['head_pose']:.1f}°")

# 运行可视化
visualize_features()