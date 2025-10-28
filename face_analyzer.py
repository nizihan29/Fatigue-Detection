import cv2
import dlib
import numpy as np
import os

class FaceAnalyzer:
    """面部关键点检测和分析"""
    
    def __init__(self, landmark_model_path=None):
        # 人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        
        # 🎯 关键点检测器 - 支持自定义路径
        if landmark_model_path and os.path.exists(landmark_model_path):
            self.predictor = dlib.shape_predictor(landmark_model_path)
            print(f"✅ 已加载关键点模型: {landmark_model_path}")
        else:
            # 尝试使用默认路径
            default_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(default_path):
                self.predictor = dlib.shape_predictor(default_path)
                print(f"✅ 已加载默认关键点模型: {default_path}")
            else:
                self.predictor = None
                print("⚠️  未找到关键点模型，将仅进行人脸检测")
    
    def detect_face_and_landmarks(self, frame):
        """检测人脸和关键点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 🎯 改进的人脸检测 - 多尺度+角度容忍
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            # 尝试调整图像
            faces = self.detector(gray, 0)  # 不进行上采样
            if len(faces) == 0:
                # 调整亮度和对比度
                enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=40)
                faces = self.detector(enhanced, 1)
        
        if len(faces) == 0:
            return None, None
        
        # 选择最大的人脸
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        if self.predictor:
            landmarks = self.predictor(gray, face)
            return face, landmarks
        else:
            return face, None
    
    def extract_fatigue_features(self, landmarks, previous_landmarks=None):
        """提取疲劳相关特征"""
        if landmarks is None:
            return None
        
        features = {}
        
        # 1. 眼睛闭合程度
        left_eye_ratio = self.eye_aspect_ratio([
            landmarks.part(36), landmarks.part(37), landmarks.part(38),
            landmarks.part(39), landmarks.part(40), landmarks.part(41)
        ])
        
        right_eye_ratio = self.eye_aspect_ratio([
            landmarks.part(42), landmarks.part(43), landmarks.part(44),
            landmarks.part(45), landmarks.part(46), landmarks.part(47)
        ])
        
        features['eye_closure'] = (left_eye_ratio + right_eye_ratio) / 2
        
        # 2. 嘴部张开程度 (打哈欠检测)
        mouth_ratio = self.mouth_aspect_ratio([
            landmarks.part(60), landmarks.part(61), landmarks.part(62), landmarks.part(63),
            landmarks.part(64), landmarks.part(65), landmarks.part(66), landmarks.part(67)
        ])
        features['mouth_openness'] = mouth_ratio
        
        # 3. 头部姿态 (简单估计)
        head_pose = self.estimate_head_pose(landmarks)
        features['head_pose'] = head_pose
        
        # 4. 头部运动 (如果提供了前一帧)
        if previous_landmarks:
            head_movement = self.calculate_head_movement(landmarks, previous_landmarks)
            features['head_movement'] = head_movement
        
        return features
    
    def eye_aspect_ratio(self, eye_points):
        """计算眼睛纵横比"""
        # 垂直距离
        A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - 
                          np.array([eye_points[5].x, eye_points[5].y]))
        B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - 
                          np.array([eye_points[4].x, eye_points[4].y]))
        
        # 水平距离
        C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - 
                          np.array([eye_points[3].x, eye_points[3].y]))
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth_points):
        """计算嘴部纵横比"""
        # 垂直距离
        A = np.linalg.norm(np.array([mouth_points[2].x, mouth_points[2].y]) - 
                          np.array([mouth_points[6].x, mouth_points[6].y]))
        
        # 水平距离
        B = np.linalg.norm(np.array([mouth_points[0].x, mouth_points[0].y]) - 
                          np.array([mouth_points[4].x, mouth_points[4].y]))
        
        mar = A / B
        return mar
    
    def estimate_head_pose(self, landmarks):
        """简单估计头部姿态"""
        # 使用鼻子和眼睛的位置关系
        nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
        left_eye_center = np.array([
            (landmarks.part(36).x + landmarks.part(39).x) / 2,
            (landmarks.part(36).y + landmarks.part(39).y) / 2
        ])
        right_eye_center = np.array([
            (landmarks.part(42).x + landmarks.part(45).x) / 2,
            (landmarks.part(42).y + landmarks.part(45).y) / 2
        ])
        
        # 简单的头部倾斜估计
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def calculate_head_movement(self, current_landmarks, previous_landmarks):
        """计算头部运动"""
        # 使用鼻子点的位移
        current_nose = np.array([current_landmarks.part(30).x, current_landmarks.part(30).y])
        previous_nose = np.array([previous_landmarks.part(30).x, previous_landmarks.part(30).y])
        
        movement = np.linalg.norm(current_nose - previous_nose)
        return movement
    
    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0)):
        """在图像上绘制关键点"""
        if landmarks is None:
            return frame
        
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, color, -1)
        
        return frame