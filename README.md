## 💾 Model Weights Download
이 프로젝트를 실행하려면 학습된 모델 가중치가 필요합니다. 아래 링크에서 다운로드하여 `src/` 또는 코드상의 경로에 위치시키세요.
* **Download Link:** [https://drive.google.com/file/d/1Vp8Oj6Tcdu-kWxWM-tOA1QeRrCsF9bNu/view?usp=sharing]

# 💊 AI 기반 경구약제 객체 탐지 및 정보 제공 시스템
## AI06 3팀 최종 프로젝트 상세 보고서

> **Date:** 2024.12.09 ~ 2024.12.17  
> **Team:** AI06 3팀  
> **Result:** Kaggle Leaderboard **mAP 0.82211 (1st Place)** > **Tech Stack:** YOLOv8, Albumentations, WBF Ensemble, Streamlit

---

## 📋 목차
1.  **프로젝트 기획 및 주제 선정**
2.  **데이터 전처리 및 탐색 (Data-Centric)**
3.  **모델 선정 및 설계 (Model-Centric)**
4.  **모델 훈련 및 성능 고도화 (Optimization)**
5.  **핵심 트러블슈팅: 0.0점의 원인과 해결 (Critical Solving)**
6.  **코드 품질 및 문서화**
7.  **결과 해석 및 비즈니스 시사점 (Service)**
8.  **예상 Q&A (Technical Defense)**
9.  **프로젝트 폴더 구조 (Project Structure)**

---

## 1. 프로젝트 기획 및 주제 선정

### 1.1 문제 정의 및 배경
* **배경:** 고령화 사회 진입으로 다제 약물 복용자가 급증하고 있습니다. 하지만 약의 생김새(원형, 타원형, 흰색 등)가 비슷하여 시각 정보만으로는 구분이 어렵습니다. 기존 서비스는 텍스트 검색 위주라 고령층 접근성이 낮고, 이미지 검색은 정확도가 떨어집니다.
* **실제 사용 시나리오 (Real-world Scenario):**
    * 사용자가 약국에서 받아온 **'약 봉지(투명 비닐)'** 상태 그대로 촬영하는 상황을 가정했습니다.
    * 이 경우 알약은 정직하게 놓여있지 않고 **360도 회전**되어 있거나, 비닐에 의한 **빛 반사**, 알약끼리 **겹쳐있는(Occlusion)** 현상이 발생합니다.
* **기술적 난이도:**
    * **Small Objects:** 알약은 이미지 전체 대비 크기가 매우 작음.
    * **Similarity:** 서로 다른 약품이지만 색상과 모양이 유사한 경우가 많음 (Inter-class similarity).

### 1.2 프로젝트 목표
* **정량적 목표:**
    * Kaggle Leaderboard 기준 **mAP 0.82 이상** 달성.
    * 서비스 가능한 수준의 **실시간 추론 (Inference Time < 0.1s)** 확보.
* **정성적 목표:**
    * **Robust Model:** 실험실 환경이 아닌, 실제 약 봉지 환경에서도 강건하게 작동하는 모델 개발.
    * **End-to-End Pipeline:** 데이터 수집부터 모델 학습, 웹 서비스(Streamlit) 구현까지 전체 파이프라인 구축.
    * **Reproducibility:** 100% 재현 가능한 코드 베이스 마련.

---

## 2. 데이터 전처리 및 탐색 (Data-Centric Approach)

### 2.1 데이터 수집 및 구조화
* **데이터셋:** AI Hub 경구약제 이미지 데이터 (Kaggle Competition Version)
* **초기 문제점 (Raw Data Issues):**
    1.  **파편화:** 수백 개의 폴더에 JSON과 이미지가 흩어져 있어 로딩 불가.
    2.  **유령 데이터:** JSON 라벨은 존재하나 실제 이미지 파일이 없는 경우 다수 발견.
    3.  **심각한 불균형:** 56개 클래스 중 55개가 50장 미만 (Long-tail Distribution). 일부는 3~5장에 불과해 학습이 불가능한 수준.

### 2.2 전처리 전략 (Preprocessing Strategy)
**1) 데이터 무결성 검사 및 병합 (Integrity Check)**
* 흩어진 JSON 파일들을 순회하며 `glob`으로 수집 후 하나의 `train.json`으로 병합하는 자동화 스크립트 작성.
* `os.path.exists()`를 통해 실제 이미지가 없는 Annotation을 영구 제거하여 학습 에러 사전 차단.
* 결과: 총 1,776장 이미지, 2,089개 객체, 56개 카테고리로 정제 완료.

**2) 상위 30종 선별 (Service Targeting)**
* 전체 56종 중 데이터가 충분하고 실제 상비약으로 쓰이는 **Top 30종**을 선정하여 `top30_pills_service.csv` DB 구축.
* 모델 학습은 56종 전체로 하되, 서비스 단에서는 30종에 집중하는 전략 수립.

### 2.3 데이터 증강 (Offline Augmentation)
단순한 데이터 뻥튀기가 아니라, **'약 봉지 환경'**을 모사하기 위해 `Albumentations`를 활용한 **Self-Contained Augmentation**을 적용했습니다.

* **Why Offline?** On-the-fly(학습 중 실시간 변환) 방식은 빠르지만, 박스 소실 여부를 눈으로 확인하기 어렵습니다. 우리는 물리적 파일로 저장하여 품질을 검수하고 **재현성**을 확보하기 위해 Offline 방식을 택했습니다.
* **증강 파이프라인 (Code Detail):**
    ```python
    pipeline = A.Compose([
        A.SafeRotate(limit=180, p=0.7), # 약 봉지 안에서 알약은 360도 회전함
        A.HorizontalFlip(p=0.5),        # 좌우 반전
        A.RandomBrightnessContrast(p=0.5), # 형광등/자연광 등 조명 변화 대응
        A.GaussianBlur(p=0.3),          # 비닐 포장으로 인한 뿌연 화질 모사
        A.GaussNoise(p=0.3)             # 카메라 노이즈 모사
    ], bbox_params=A.BboxParams(format='coco', min_visibility=0.3))
    ```
* **Retry Logic:** 증강 후 Bbox가 이미지 밖으로 나가 사라지는 경우를 방지하기 위해, 박스가 보존될 때까지 재시도하는 로직을 추가 구현했습니다.
* **성과:** 모든 클래스 최소 50장 이상 확보 (총 2,400+장 구축).

---

## 3. 모델 선정 및 설계 (Model Architecture)

### 3.1 모델 선정 과정 (Comparative Study)
세 가지 모델을 선정하여 장단점을 비교 분석했습니다.

| 모델 | 아키텍처 | 선정 이유 | 예상 강점 | 예상 약점 |
| :--- | :--- | :--- | :--- | :--- |
| **Faster R-CNN** | ResNet50 + FPN | 2-stage Detector의 정석, 안정성 | 높은 정확도 | 느린 추론 속도 |
| **Detectron2** | R50-FPN 3x | SOTA 프레임워크, 강력한 성능 | 커스터마이징 용이 | 설정 복잡도 |
| **YOLOv8x** | CSPDarknet + PAN | **최신 1-stage, 실시간성** | **속도/정확도 균형** | 작은 객체 탐지력 |

**👉 최종 선정: YOLOv8x (Extra Large)**
* 이유 1: **실시간성.** 서비스 적용을 위해 0.1초 이내의 응답 속도가 필수적임.
* 이유 2: **최신 아키텍처.** Anchor-free 방식과 Mosaic Augmentation 내장으로 학습 효율이 높음.
* 이유 3: **성능.** 실험 결과 mAP가 가장 우수했음.

### 3.2 모델 설계 디테일
* **Input Size:** 640x640 (YOLO 표준, 과적합 방지 및 속도 최적화)
* **Batch Size:** 16 (Colab T4 GPU 메모리 한계 고려)
* **Optimizer:** AdamW (일반화 성능 강화)
* **Scheduler:** CosineAnnealingLR (Local Minima 탈출 유도)
* **Loss Function:** CIoU Loss (Box 정확도) + DFL (Distribution Focal Loss)

---

## 4. 모델 훈련 및 성능 고도화 (Optimization)

### 4.1 실험 결과 요약
* **Faster R-CNN:** mAP 0.77313 (Baseline으로 준수함)
* **Detectron2:** mAP 0.68491 (해상도 800px 과다 설정으로 인한 노이즈 오탐 문제 발생)
* **YOLOv8x (Base):** mAP 0.82073 (Best Single Model)

### 4.2 성능 개선: WBF 앙상블 (Ensemble Strategy)
단일 모델의 성능 한계를 넘기 위해 **Weighted Boxes Fusion (WBF)** 기법을 도입했습니다.

* **문제:** 일반적인 NMS(Non-Maximum Suppression)는 겹치는 박스 중 점수가 높은 하나만 남기고 나머지는 버립니다. 이는 정보의 손실입니다.
* **해결:** WBF는 겹치는 박스들의 좌표를 **가중 평균(Weighted Average)**하여 더 정확한 위치를 찾아냅니다.
* **구현 조합:**
    1.  `YOLOv8x (Standard)`: 기본 추론 (정확도 중시) -> **가중치 2**
    2.  `YOLOv8x (TTA)`: Test Time Augmentation 적용 (Recall 중시) -> **가중치 1**
* **성과:** 단일 모델 대비 **+0.78%** 향상된 **mAP 0.82211** 달성 (최종 1위).

---

## 5. 핵심 트러블슈팅: 0.0점의 원인과 해결 (Critical Solving)

프로젝트 막바지에 발생한 치명적인 이슈와 이를 해결한 과정입니다.

### 5.1 문제 상황 (The "0.00000" Incident)
* **현상:** 모델 학습 mAP는 0.82가 넘는데, 캐글 제출 시 점수가 **0.00000**이 나오는 현상 발생.
* **원인 분석:**
    1.  **Image ID:** 확인 결과 파일명과 CSV ID는 일치함 (정상).
    2.  **Category ID (범인):**
        * YOLO 모델은 학습 시 클래스를 `0 ~ 55`의 인덱스(Index)로 변환하여 학습합니다.
        * 하지만 실제 대회 정답 파일(Ground Truth)은 `1899`, `2482` 등의 **고유 약품 코드(Unique Code)**를 요구했습니다.
        * 즉, 우리는 "0번 약입니다"라고 제출했고, 서버는 "1899번 약을 찾아라"라고 채점하여 0점이 된 것입니다.

### 5.2 해결책: 하드코딩 매핑 (Hard-coded Mapping)
* **시도 1 (실패):** `train.json`을 읽어서 동적으로 매핑하려 했으나, 파일 정렬 순서에 따라 인덱스가 꼬이는 불안정성 발견.
* **시도 2 (성공 - Smart Fix):** 데이터의 불확실성을 배제하기 위해, **사람이 직접 검증한 매핑 테이블**을 코드에 박아넣는(Hard-coding) 방식을 채택했습니다.
    ```python
    # 0.0점 탈출의 핵심 코드
    CATEGORY_MAPPING = {
        0: 1899, 1: 2482, ... 55: 41767
    }
    real_id = CATEGORY_MAPPING[int(yolo_pred)]
    ```
* **결과:** 점수가 즉시 **0.82211로 복구**되었으며, 시스템의 **안정성(Stability)**을 확보했습니다.

---

## 6. 코드 품질 및 문서화

### 6.1 모듈화 및 재현성
* **OOP 구조:** `Config`, `Dataset`, `Trainer` 클래스로 분리하여 유지보수성 향상.
* **Seed Fixing:** `seed_everything(42)` 함수를 통해 누가 실행해도 동일한 결과를 보장.
* **경로 관리:** Google Drive와 Colab Local 데이터를 분리하여 I/O 병목 현상 해결.

### 6.2 협업을 위한 문서화
* 팀원 간 공유를 위해 **상세한 주석**과 **Markdown 가이드**를 Notebook 내에 포함.
* **버전 관리:** `v9`, `backup` 등 파일명에 버전을 명시하고 `requirements.txt`로 환경 통일.

---

## 7. 결과 해석 및 비즈니스 시사점 (Service Application)

### 7.1 기술적 성과 해석
* **YOLO의 승리:** 복잡한 2-stage 모델보다, 최신 증강 기법이 적용된 1-stage 모델이 소형 객체 탐지에서도 충분히 강력함을 입증했습니다.
* **데이터의 중요성:** 모델 아키텍처를 바꾸는 것보다, **양질의 데이터(Offline Augmentation)**를 확보하는 것이 성능 향상에 훨씬 큰 영향을 미쳤습니다.

### 7.2 비즈니스 모델: "나만의 AI 약사" (Streamlit)
단순 탐지를 넘어, 실제 사용자를 위한 웹 서비스 프로토타입을 기획했습니다.
* **기능:** 사용자가 알약 사진을 찍어 업로드.
* **프로세스:**
    1.  YOLO 모델이 알약 탐지 및 ID 추출.
    2.  추출된 ID를 `drug_info.csv` DB와 매핑.
    3.  화면에 **약품명, 효능, 복용법, 주의사항**을 출력.
* **기대 효과:** 시각적 구분이 어려운 노년층의 오남용 방지 및 약국 업무 효율화.

---

## 8. 예상 Q&A (Technical Defense)

**Q1. 왜 Detectron2 성능이 가장 낮았나요?**
> A: "높은 해상도가 항상 정답은 아니다"는 교훈을 얻었습니다. Detectron2의 입력 크기를 800px로 높게 설정했는데, 오히려 배경의 노이즈(먼지, 비닐 주름)를 소형 객체로 오인하는 과다 탐지(False Positive) 문제가 발생했습니다.

**Q2. TTA(Test Time Augmentation)를 적용했는데 왜 점수가 떨어졌나요?**
> A: TTA는 양날의 검입니다. 이미지를 회전/반전 시키는 과정에서 일부 박스가 화면 밖으로 밀려나거나, 대칭형 알약의 경우 좌우 반전 시 혼란을 주기도 했습니다. 하지만 단독 사용이 아닌 **WBF 앙상블** 재료로 사용했을 때는 모델의 다양성(Diversity)을 높여주어 최종 성능 향상에 기여했습니다.

**Q3. 프로젝트에서 가장 어려웠던 점은?**
> A: **데이터 불균형**과 **0.0점 이슈**였습니다.
> 1. 50장 미만의 클래스가 대부분인 상황을 타개하기 위해 `SafeRotate`와 `Retry Logic`을 직접 구현한 오프라인 증강을 적용했습니다.
> 2. 학습 인덱스와 제출 ID가 달라서 발생한 0점 문제는, 외부 변수를 차단한 **하드코딩 매핑**을 통해 가장 확실하고 안전하게 해결했습니다.

**Q4. 향후 개선 방향은?**
> A: 라벨링이 누락된 419장의 이미지를 발견했습니다. 현재 0.82점 모델을 이용해 이 데이터에 **Pseudo-labeling**을 수행하고 재학습시킨다면, 0.85점 이상으로 성능을 높일 수 있을 것입니다.

**Q5. 왜 YOLOv8x를 두 개 앙상블 했나요? (중복 질문 아님)**
> A: 실험 결과를 보시면:
> * Faster R-CNN + YOLO: 0.805 (Faster가 발목)
> * Detectron2 + YOLO: 0.793 (Detectron이 발목)
> * **YOLO(No-TTA) + YOLO(TTA): 0.822 (최고)**
>
> **결론:** 성능이 떨어지는 모델을 섞으면 오히려 평균이 깎입니다.
> 같은 모델이지만 추론 방식이 다른 경우(No-TTA vs TTA)가 **다양성과 균형을 동시에** 확보할 수 있었습니다.

---

## 9. 프로젝트 폴더 구조 (Project Structure)

`v9` 코드 베이스 기준의 최종 폴더 구조입니다.

```text
Project_Root/
│
├── data/                              # 데이터셋 (Colab 로컬 ./data)
│   ├── train_images/                  # 원본 및 증강된 이미지 (2,400+장)
│   ├── train.json                     # 병합된 원본 라벨
│   └── train_final_augmented.json     # 오프라인 증강이 적용된 최종 학습용 라벨
│
├── models/                            # 모델 저장소
│   ├── yolov8_best.pt                 # 학습 완료된 YOLOv8 Best Weight
│   └── yolov8x.pt                     # Pre-trained Base Weight
│
├── submission/                        # 결과 제출물
│   ├── submission_ensemble_WBF.csv    # WBF 앙상블 적용 결과
│   └── submission_FINAL_FIXED.csv     # ID 매핑(0.0점 해결) 적용된 최종 제출 파일
│
├── app/                               # 서비스 데모 (Streamlit)
│   ├── streamlit_app.py               # 웹 어플리케이션 구동 코드
│   └── drug_info.csv                  # 약품 정보 DB (ID, 효능, 주의사항 매핑)
│
├── AI06_3팀_프로젝트_v9.ipynb          # 전체 파이프라인 실행 노트북 (Main)
└── requirements.txt                   # 필요 라이브러리 (albumentations, ultralytics 등)

Copyright © 2025 AI06 Team 3. All Rights Reserved.

---

**## 프로젝트 협업일지**
김재혁: 
김민주: 
김민철: 
이승준: 
