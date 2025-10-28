from collections import deque
import numpy as np
from face_analyzer import FaceAnalyzer

class RealTimeFatigueDetector:
    def __init__(self, landmark_model_path=None):
        # 🎯 面部分析器 - 规则引擎核心
        self.face_analyzer = FaceAnalyzer(landmark_model_path)
        
        # 🎯 状态跟踪
        self.landmark_buffer = deque(maxlen=5)  # 保存最近5帧的关键点
        self.eye_closure_history = deque(maxlen=30)  # 眼睛闭合历史
        self.mouth_openness_history = deque(maxlen=30)  # 嘴部张开历史
        self.state_history = deque(maxlen=10)  # 状态历史
        self.fatigue_scores = deque(maxlen=50)  # 疲劳分数历史
        
        # 🎯 疲劳检测阈值（可调整）
        self.EYE_CLOSURE_THRESHOLD = 0.20  # 眼睛闭合阈值
        self.MOUTH_OPEN_THRESHOLD = 0.55   # 嘴部张开阈值
        self.HEAD_POSE_THRESHOLD = 12      # 头部姿态阈值
        self.FATIGUE_SCORE_THRESHOLD = 55  # 疲劳分数阈值
        
        print("✅ 基于规则的疲劳检测器初始化完成")
    
    def detect_face_robust(self, frame):
        """鲁棒的人脸检测"""
        return self.face_analyzer.detect_face_and_landmarks(frame)
    
    def analyze_fatigue_indicators(self, landmarks, previous_landmarks):
        """分析疲劳指标"""
        if landmarks is None:
            return None
        
        features = self.face_analyzer.extract_fatigue_features(landmarks, previous_landmarks)
        if not features:
            return None
        
        fatigue_score = 0
        
        # 1. 眼睛闭合检测
        eye_closure = features['eye_closure']
        if eye_closure < self.EYE_CLOSURE_THRESHOLD:
            fatigue_score += 40
            self.eye_closure_history.append(1)
        else:
            self.eye_closure_history.append(0)
        
        # 2. 连续闭眼检测
        if len(self.eye_closure_history) >= 10:
            recent_closures = list(self.eye_closure_history)[-10:]
            closure_rate = sum(recent_closures) / len(recent_closures)
            if closure_rate > 0.6:  # 60%的时间闭眼
                fatigue_score += 25
        
        # 3. 打哈欠检测
        mouth_openness = features['mouth_openness']
        if mouth_openness > self.MOUTH_OPEN_THRESHOLD:
            fatigue_score += 35
            self.mouth_openness_history.append(1)
        else:
            self.mouth_openness_history.append(0)
        
        # 4. 连续打哈欠检测
        if len(self.mouth_openness_history) >= 15:
            recent_yawns = list(self.mouth_openness_history)[-15:]
            yawn_rate = sum(recent_yawns) / len(recent_yawns)
            if yawn_rate > 0.4:  # 40%的时间打哈欠
                fatigue_score += 15
        
        # 5. 头部姿态异常
        head_pose = features['head_pose']
        if abs(head_pose) > self.HEAD_POSE_THRESHOLD:
            fatigue_score += 20
        
        # 6. 头部运动检测（如果可用）
        if 'head_movement' in features and features['head_movement'] > 5:
            fatigue_score += 10
        
        # 限制分数在0-100之间
        fatigue_score = min(fatigue_score, 100)
        
        return fatigue_score, features
    
    def predict_fatigue(self, frame):
        """预测疲劳状态"""
        face, landmarks = self.detect_face_robust(frame)
        
        if face is None:
            return 0, 0.0, None, None
        
        # 🎯 获取前一帧关键点用于时序分析
        previous_landmarks = self.landmark_buffer[-1] if self.landmark_buffer else None
        
        # 🎯 使用规则引擎计算疲劳分数
        fatigue_info = self.analyze_fatigue_indicators(landmarks, previous_landmarks)
        
        # 🎯 保存当前关键点
        self.landmark_buffer.append(landmarks)
        
        if fatigue_info:
            fatigue_score, features = fatigue_info
            self.fatigue_scores.append(fatigue_score)
            
            # 使用滑动平均平滑疲劳分数
            if len(self.fatigue_scores) > 0:
                smoothed_score = np.mean(list(self.fatigue_scores))
            else:
                smoothed_score = fatigue_score
            
            # 判断疲劳状态
            if smoothed_score > self.FATIGUE_SCORE_THRESHOLD:
                state = 1  # 疲劳
                confidence = min(smoothed_score / 100.0, 0.95)
            else:
                state = 0  # 清醒
                confidence = 1.0 - (smoothed_score / 100.0)
            
            self.state_history.append((state, confidence))
            return state, confidence, features, face
        
        return 0, 0.5, None, face
    
    def get_smoothed_prediction(self):
        """获取平滑后的预测结果"""
        if not self.state_history:
            return 0, 0.0
        
        states = [state for state, _ in self.state_history]
        confidences = [conf for _, conf in self.state_history]
        
        # 使用多数投票决定状态
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        predicted_state = max(state_counts, key=state_counts.get)
        avg_confidence = np.mean(confidences)
        
        return predicted_state, avg_confidence