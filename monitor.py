import cv2
import time
import numpy as np
from collections import deque
from fatigue_detector import RealTimeFatigueDetector

class FatigueMonitor:
    def __init__(self, landmark_model_path=None):
        self.detector = RealTimeFatigueDetector(landmark_model_path)
        self.performance_stats = {
            'frame_count': 0,
            'detection_times': deque(maxlen=100),
            'fatigue_frames': 0
        }
    
    def run_detection(self, camera_id=0, show_landmarks=True):
        """运行实时检测"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Starting real-time fatigue monitoring...")
        print("Press 'q' to quit")
        print("Press 'r' to reset statistics")
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 480))
            self.performance_stats['frame_count'] += 1
            
            # 进行疲劳检测
            state, confidence, features, face = self.detector.predict_fatigue(frame)
            
            # 更新统计信息
            if state == 1:
                self.performance_stats['fatigue_frames'] += 1
            
            # 获取平滑后的预测结果
            smoothed_state, avg_confidence = self.detector.get_smoothed_prediction()
            
            # 绘制结果
            frame = self.draw_detection_results(frame, smoothed_state, avg_confidence, 
                                              features, face, show_landmarks)
            
            # 计算并显示性能信息
            detection_time = time.time() - start_time
            self.performance_stats['detection_times'].append(detection_time)
            fps = 1.0 / detection_time if detection_time > 0 else 0
            
            self.draw_performance_info(frame, fps)
            
            cv2.imshow('Fatigue Monitoring', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_statistics()
        
        cap.release()
        cv2.destroyAllWindows()
        self.print_final_statistics()
    
    def draw_detection_results(self, frame, state, confidence, features, face, show_landmarks):
        """在帧上绘制检测结果"""
        label = "FATIGUE" if state == 1 else "ALERT"
        color = (0, 0, 255) if state == 1 else (0, 255, 0)
        
        # 绘制状态和置信度
        cv2.putText(frame, f"State: {label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 绘制人脸框
        if face is not None:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # 绘制关键点
            if show_landmarks and hasattr(self.detector.face_analyzer, 'predictor'):
                landmarks = self.detector.landmark_buffer[-1] if self.detector.landmark_buffer else None
                if landmarks:
                    frame = self.detector.face_analyzer.draw_landmarks(frame, landmarks, color)
        
        # 显示疲劳特征
        if features:
            y_offset = 110
            feature_texts = [
                f"Eye Closure: {features['eye_closure']:.3f}",
                f"Mouth Openness: {features['mouth_openness']:.3f}",
                f"Head Pose: {features['head_pose']:.1f} deg"
            ]
            
            if 'head_movement' in features:
                feature_texts.append(f"Head Movement: {features['head_movement']:.2f}")
            
            for text in feature_texts:
                cv2.putText(frame, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        return frame
    
    def draw_performance_info(self, frame, fps):
        """绘制性能信息"""
        fatigue_ratio = (self.performance_stats['fatigue_frames'] / 
                        self.performance_stats['frame_count'])
        
        info_texts = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.performance_stats['frame_count']}",
            f"Fatigue Ratio: {fatigue_ratio:.2f}"
        ]
        
        y_offset = 200
        for text in info_texts:
            cv2.putText(frame, text, (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def reset_statistics(self):
        """重置统计信息"""
        self.performance_stats = {
            'frame_count': 0,
            'detection_times': deque(maxlen=100),
            'fatigue_frames': 0
        }
        self.detector.fatigue_scores.clear()
        self.detector.state_history.clear()
        print("Statistics reset!")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        print("\n=== Final Statistics ===")
        print(f"Total frames processed: {self.performance_stats['frame_count']}")
        print(f"Fatigue frames: {self.performance_stats['fatigue_frames']}")
        
        if self.performance_stats['frame_count'] > 0:
            fatigue_ratio = (self.performance_stats['fatigue_frames'] / 
                           self.performance_stats['frame_count'])
            print(f"Fatigue ratio: {fatigue_ratio:.3f}")
        
        if self.performance_stats['detection_times']:
            avg_time = np.mean(list(self.performance_stats['detection_times']))
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Average FPS: {avg_fps:.1f}")


def main():
    # 初始化监测器
    # 注意：需要下载dlib的68点关键点模型
    landmark_model_path = "./models/shape_predictor_68_face_landmarks.dat"  # 修改为您的模型路径
    
    monitor = FatigueMonitor(landmark_model_path)
    
    # 开始实时监测
    monitor.run_detection(camera_id=0, show_landmarks=True)

if __name__ == "__main__":
    main()