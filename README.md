# level1_bookratingprediction-recsys-02
# 1-1. 프로젝트 개요

- 프로젝트 주제
    - 과거 특정 사용자가 어떠한 책에 대해 내린 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 테스크
    - 진행기간 : 4월 10일 (월) 10:00 ~ 4월 20일 (목) 19:00
- 프로젝트 구현 내용
    - 어떠한 소비자가 수백 페이지로 이루어진 책을 상세히 읽지 않아도 본인의 과거 의사결정 정보를 통해 학습한 머신러닝/딥러닝 모델으로 책의 선호도를 예측해 책 구매 결정에 대한 도움을 줄 수 있는 프로세스를 개발함
- 활용 장비 및 재료(개발환경, 협업 tool 등)
    - (팀 구성 및 컴퓨팅 환경) 4인 1팀, 인당 V100 서버를 VSCode와 SSH로 연결하여 사용
    - (협업 환경) GitHub, Notion
    - (의사 소통) Slack, Zoom
- 프로젝트 구조 및 사용 데이터셋의 구조도(연관도)
    
    ```
    ├── [1] main.py
    ├── [2] src
    │   ├── data
    │   │   └── dl_data.py
    │   ├── models
    │   │   ├── DCN(= Parallel DCN with Context)
    │   ├── train
    │   │   └── trainer.py
    │   └── utils.py
    ├── [3] Data_Search_Final.ipynb
    ├── [4] Data_Search_Prev.ipynb
    ├── [5] Make_ensembles.ipynb
    ├── [6] Optuna-Catboost-10Fold.ipynb
    └── [7] Optuna-Total-DCNC-10Fold.ipynb
    ```
    

# 1-2. 프로젝트 팀 구성 및 역할

- 김수민_T5040: EDA 및 Preprocessing, FM/FFM/NCF 성능 확인 및 튜닝, Rule-based 모델 설계
- 김영서_T5042: EDA 및 Preprocessing, DCN/FM/FFM 성능 확인
- 김지우_T5063: EDA 및 Preprocessing, DCN/WDN 성능 확인 및 튜닝, Rule-Based/DeepCoNN 모델 설계
- 박수현_T5085:  팀의 목표 설정 및 리딩, EDA 및 Preprocessing, Catboost 설계 및 튜닝, Parallel DCN with Context 설계 및 튜닝

# 1-3. 프로젝트 수행 절차 및 방법

<전반적인 Process>

![Untitled](https://user-images.githubusercontent.com/71757471/234790327-8d894529-81cc-4501-8b7a-c85008c9aa52.png)

# 1-4. 프로젝트 수행 결과

- 탐색적 분석 및 전처리
    - Fill The Missing Value
    - Cardinality Reduction
    - Binning
    
- 모델 개요
    - **Catboost**
    - **PDCNC(Parallel DCN with Context)**
        - Baseline Code에서 수정한 점
            - Parallel DCN : High-Order Interaction을 잡아낼 수 있는 Cross Network와 비선형적인 상호작용을 잡아낼 수 있는 MLP의 조합을 기대하여 Stacked DCN에서 Parallel DCN으로 구조를 변경하였습니다.
            - Context Information 활용 : User-Book 간에 Interaction 뿐만 아니라 User Profile, Book Profile 정보를 활용하면 더욱더 풍부한 예측이 이루어질 것으로 기대하였습니다.
            - 그 밖에 Model Capacity를 증가시켜 약간의 성능 향상을 이끌어냈습니다.
- 모델 선정 및 분석
    - Catboost
        - 대부분 Categorical Feature로 이루어졌기 때문에 이러한 부분을 Target Encoding으로 대응할 수 있는 Catboost 모델을 사용하였습니다.
        - 아래에 언급한 그림과 같이 group(user_id).var()과 group(book_author).var()의 경우, 0에 근접한 분포를 보이는 것을 알 수 있습니다. 즉, 특정 유저 또는 책을 작성한 저자에 따른 평점의 경향성이 존재함을 의미 하는 것으로, Catboost에서 제공하는 Target Encoding이 이러한 경향성을 담아낼 수 있을 것으로 기대하여 Catboost를 사용하였습니다.
        
        ![Untitled](https://user-images.githubusercontent.com/71757471/234790282-2323bbac-8fb5-4567-8e05-1dc11f250f1a.png)
        
    - PDCNC
        - Baseline으로 제공되는 DCN에 Context Information을 추가하여 딥러닝 모델을 구성하였습니다. 그래서 Context Information을 사용하는 그밖에 딥러닝 모델들인 FM, FFM, WDN과 비교하여 아래에 언급한 그림처럼 PDCNC이 월등한 성능을 보여 PDCNC을 사용하였습니다.
            
            ![Untitled](https://user-images.githubusercontent.com/71757471/234790298-89f2ca9f-a6a9-430f-aed2-1da578f0e52c.png)
            
- 모델 평가 및 개선
    - 모델 평가
        - Catboost : Cross Validation   
        - PDCNC : Holdout(train_test_split(ratio = 0.2))

            
    - 모델 개선
        - Each Single Model
            
            ![Untitled](https://user-images.githubusercontent.com/71757471/234790314-3dab2715-abe8-4e95-ab6a-af0fdf241f2d.png)
            
            1. N-Stratified Fold를 이용해 1개의 Fold는 Validation Fold, 나머지는 Train Fold로 지정해 총 N번의 반복이 이루어집니다. 
            2. 각 반복에서 Optuna로 Hyper Parameter Tuning이 이루어지고 학습된 N-Estimators에 대해 Voting으로 모델을 개선했습니다. 
        
        - 최종적으로 학습된 모델(→Ensemble)
            - Weighted Summation of Models : 0.6 * Catboost + 0.4 PDCNC
    
- 시연 결과(모델 성능)
    - Single Model(Catboost) Prediction
        
        ![Untitled](https://user-images.githubusercontent.com/71757471/234790316-aec3f40b-609a-4475-b575-dc828bfd0969.png)
        
    - Ensemble(0.6 * Catboost + 0.4 * PDCNC) Prediction
        
        ![Untitled](https://user-images.githubusercontent.com/71757471/234790320-123712b9-0145-41c1-9a04-f0a9d3ada95c.png)
