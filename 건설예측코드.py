import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

### train ,test 셋 불러오기
test = pd.read_csv('/test.csv')
train = pd.read_csv('/train.csv')



## test셋과 train 셋의 데이터 컬럼의 차이가 있어 test셋에 맞춰 진행
train1 = train[test.columns]
train1['Y_LABEL'] = train['Y_LABEL']

# 범주형 변수 변환
train1['COMPONENT_ARBITRARY'] = train1['COMPONENT_ARBITRARY'].astype('category').cat.codes
test['COMPONENT_ARBITRARY'] = test['COMPONENT_ARBITRARY'].astype('category').cat.codes

x = train1.drop(['ID','Y_LABEL'] ,axis = 1)
y = train1['Y_LABEL']

x_train,x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 ,stratify=y , random_state=42)


## SMOTE 오버샘플링 진행
smote = SMOTE(random_state=0)
X_train_over,y_train_over = smote.fit_resample(x_train,y_train)

model = RandomForestClassifier(random_state=42)


## 그리드서치 진행
param_grid = {
    'criterion': ['gini', 'entropy'],
    'n_estimators':[10,50,100,200,500],
    'max_depth': [5, 10, 50,100],
    'min_samples_leaf' :[1,2,5,10]
}

grid_search = GridSearchCV(model, param_grid=param_grid) 
grid_search.fit(X_train_over,y_train_over)
print(grid_search.best_params_)
print(grid_search.best_score_)


## 그리트 서치 이후 나온 파라미터로 변경
model = RandomForestClassifier(criterion= 'gini',max_depth= 50, min_samples_leaf= 1, n_estimators= 500,random_state=42)
model.fit(X_train_over,y_train_over)

## 모델 예측
y_pred = model.predict(x_test)
print(f1_score(y_test , y_pred , average = 'macro'))
test_id = test['ID']
test = test.drop(['ID'] , axis = 1)
y_pred = model.predict(test)


# 결과 파일 제출
df = pd.DataFrame({'ID':test_id , 'Y_LABEL':y_pred})
df.to_csv('sda2.csv' , index = False)

