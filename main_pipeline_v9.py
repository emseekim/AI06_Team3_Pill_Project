# -*- coding: utf-8 -*-
"""
💊 AI06 3팀 Final Project: Main Pipeline (v9)
=============================================
Description: YOLOv8 Inference + WBF Ensemble + Hard-coded Mapping
Author: AI06 Team 3
Date: 2024.12.18
"""

import os
import cv2
import re
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ==========================================
# 1. Configuration & Hard-coded Mapping
# ==========================================
class Config:
    SEED = 42
    IMG_SIZE = 640
    CONF_THR = 0.01
    IOU_THR = 0.55
    SKIP_BOX_THR = 0.01
    
    # 경로 설정 (사용자 환경에 맞게 수정 필요)
    MODEL_PATH = '/content/drive/MyDrive/My_AI_Project/20251212_backup/YOLO/yolov8_best.pt'
    TEST_DIR = './data/test_images'
    OUTPUT_DIR = '/content/drive/MyDrive/My_AI_Project/Final_Project_Package_20251215'
    OUTPUT_FILE = 'submission_FINAL_FIXED.csv'

# 0.0점 탈출을 위한 핵심 매핑 테이블
CATEGORY_MAPPING = {
    0: 1899, 1: 2482, 2: 3350, 3: 3482, 4: 3543, 5: 3742, 6: 3831, 7: 4542, 8: 12080,
    9: 12246, 10: 12777, 11: 13394, 12: 13899, 13: 16231, 14: 16261, 15: 16547,
    16: 16550, 17: 16687, 18: 18146, 19: 18356, 20: 19231, 21: 19383, 22: 19606,
    23: 19860, 24: 20013, 25: 20237, 26: 20876, 27: 21324, 28: 21770, 29: 22073,
    30: 22346, 31: 22361, 32: 24849, 33: 25366, 34: 25437, 35: 25468, 36: 27732,
    37: 27776, 38: 27925, 39: 27992, 40: 29122, 41: 29344, 42: 29450, 43: 29666,
    44: 30307, 45: 31862, 46: 31884, 47: 32309, 48: 33008, 49: 33207, 50: 33879,
    51: 34597, 52: 35205, 53: 36636, 54: 38161, 55: 41767
}

# ==========================================
# 2. Utils
# ==========================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_image_id(filename):
    """파일명에서 숫자만 추출 (예: test_0123.jpg -> 123)"""
    numbers = re.findall(r'\d+', filename)
    return int(''.join(numbers)) if numbers else None

# ==========================================
# 3. Main Logic (Inference & Ensemble)
# ==========================================
def run_inference():
    print("="*60)
    print("🚀 [Start] YOLOv8 Inference Pipeline")
    print("="*60)
    
    seed_everything(Config.SEED)
    
    # 1. 모델 로드
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Error: Model not found at {Config.MODEL_PATH}")
        return
    
    model = YOLO(Config.MODEL_PATH)
    print(f"✅ Model Loaded: {Config.MODEL_PATH}")
    
    # 2. 이미지 리스트 확보
    test_imgs = glob.glob(os.path.join(Config.TEST_DIR, '*.jpg')) + \
                glob.glob(os.path.join(Config.TEST_DIR, '*.png'))
    print(f"✅ Found {len(test_imgs)} images in {Config.TEST_DIR}")
    
    if len(test_imgs) == 0:
        print("❌ Error: No images found. Please check data directory.")
        return

    results_data = []
    annotation_id = 1
    
    # 3. 추론 루프 (Tqdm)
    for img_path in tqdm(test_imgs, desc="Inference"):
        try:
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            img_id = get_image_id(os.path.basename(img_path))
            if img_id is None: continue
            
            # --- Ensemble Strategy ---
            # 1) Standard Prediction (Weight: 2)
            pred1 = model.predict(img_path, conf=Config.CONF_THR, imgsz=Config.IMG_SIZE, 
                                augment=False, verbose=False)[0]
            
            # 2) TTA Prediction (Weight: 1)
            pred2 = model.predict(img_path, conf=Config.CONF_THR, imgsz=Config.IMG_SIZE, 
                                augment=True, verbose=False)[0]
            
            boxes_list = []
            scores_list = []
            labels_list = []
            
            # Normalize Coordinates
            for pred in [pred1, pred2]:
                if len(pred.boxes) == 0:
                    boxes_list.append([]); scores_list.append([]); labels_list.append([])
                    continue
                
                boxes = pred.boxes.xyxy.cpu().numpy().copy()
                boxes[:, [0, 2]] /= w_img
                boxes[:, [1, 3]] /= h_img
                boxes = np.clip(boxes, 0, 1)
                
                boxes_list.append(boxes.tolist())
                scores_list.append(pred.boxes.conf.cpu().numpy().tolist())
                labels_list.append(pred.boxes.cls.cpu().numpy().tolist())
            
            # WBF Ensemble
            if any(len(b) > 0 for b in boxes_list):
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list,
                    weights=[2, 1], 
                    iou_thr=Config.IOU_THR, 
                    skip_box_thr=Config.SKIP_BOX_THR
                )
                
                # Save Results
                for box, score, label in zip(boxes, scores, labels):
                    # Denormalize
                    x1 = box[0] * w_img
                    y1 = box[1] * h_img
                    x2 = box[2] * w_img
                    y2 = box[3] * h_img
                    
                    # Clipping
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    w, h = x2 - x1, y2 - y1
                    
                    if w < 5 or h < 5: continue
                    
                    # ⭐ Hard-coded Mapping (Index -> Real ID)
                    yolo_idx = int(label)
                    real_id = CATEGORY_MAPPING.get(yolo_idx, -1)
                    
                    if real_id != -1:
                        results_data.append({
                            'annotation_id': annotation_id,
                            'image_id': img_id,
                            'category_id': real_id,
                            'bbox_x': float(x1),
                            'bbox_y': float(y1),
                            'bbox_w': float(w),
                            'bbox_h': float(h),
                            'score': float(score)
                        })
                        annotation_id += 1
                        
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            continue

    # 4. 결과 저장
    if results_data:
        df = pd.DataFrame(results_data)
        # Type Casting
        for col in ['annotation_id', 'image_id', 'category_id']:
            df[col] = df[col].astype(int)
            
        # Ensure Output Dir
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILE)
        
        df.to_csv(save_path, index=False)
        print("\n" + "="*60)
        print(f"🎉 Success! Submission file saved at:")
        print(f"📂 {save_path}")
        print(f"📊 Total Detections: {len(df)}")
        print("="*60)
    else:
        print("\n❌ Failed: No predictions made.")

if __name__ == "__main__":
    run_inference()