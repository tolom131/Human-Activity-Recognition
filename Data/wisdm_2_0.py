import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np
from scipy import stats

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def read_data(self):
        column_names = ['user-id',  # 1
                        'activity', # 2
                        'timestamp',# 3
                        'x-axis',   # 4
                        'y-axis',   # 5
                        'z-axis']   # 6

        df = pd.read_csv(self.file_path,
                         header=None,
                         names=column_names)
    
        df['z-axis'].replace(regex=True,
                             inplace=True,
                             to_replace=r';',
                             value=r'')
        df['z-axis'] = df['z-axis'].apply(self.convert_to_float)
        
        
        df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
        df.dropna(axis=0, how='any', inplace=True)
        LABEL = 'ActivityEncoded'
        
        # abelEncoder를 통해 레이블의 문자열을 정수로 변환)
        le = preprocessing.LabelEncoder()
        df[LABEL] = le.fit_transform(df['activity'].values.ravel())
        
        num_classes = le.classes_.size      
        return df, num_classes
    
    def convert_to_float(self, x):
        try:
            return np.float(x)
        except:
            return np.nan
        
    def create_segments_and_labels(self, df):
        time_steps = 200
        step = 200
        num_classes = 6
        n_features = 3
        
        segments = []
        labels = []
        id_user = []
        
        # segment는 이중 리스트인데, segment = [x축 가속도, y축 가속도, z축 가속도, 해당하는 레이블(activity), id_user]가 타입 스텝만큼 저장되어 있다.
        # label_data는 이러한 이중 리스트를 계속 추가하는데, Downstair은 0이므로 0번째 인덱스에 이중 리스트가 담긴다.
        # 그러면 각 label_data는 삼중 리스트로 (batch_size, TIME_STEP, 5)만큼 저장되어 있을 것이다.
        label_data = [[] for _ in range(num_classes)]
            
        for i in range(0, len(df) - time_steps, step):
            xs = df['x-axis'].values[i: i + time_steps]
            ys = df['y-axis'].values[i: i + time_steps]
            zs = df['z-axis'].values[i: i + time_steps]
            
            if(max(xs) > 20 or max(ys) > 20 or max(zs) > 20):
                continue

            label = stats.mode(df["ActivityEncoded"][i: i + time_steps])
            user = stats.mode(df['user-id'][i: i + time_steps])
                               
            # label example : ModeResult(mode=array([5]), count=array([200]))
            if(time_steps == label[1][0]):
                # label[0][0]은 숫자다.반면 xs, ys, zs는 행단위로 쌓아올린 1차원 데이터이므로 같이 쌓아주기 위해선 time_step의 길이만큼 label[0][0]을 늘려주어야 한다.
                # user도 마찬가지이다.
                extended_label = np.full((time_steps, 1), label[0][0])
                extended_user = np.full((time_steps, 1), user[0][0])
                segment = np.column_stack([xs, ys, zs, extended_label, extended_user])
                label_data[label[0][0]].append(segment)
        
        # 본 구현에서는 모든 레이블에 대한 data를 합쳐서 하나의  dataset으로 만든다.
        # 3차원 데이터에 대해서, 큐브를 예시로 든다면 큐브를 하나 두고 그 밑에 새로운 큐브를 두는 방식으로 (MIN_LABEL*6, TIMESTEP, 5)로 만든다.
        extracted_label_data = label_data[0]
        for i in range(1, num_classes):
            extracted_label_data = np.concatenate((extracted_label_data, label_data[i]), axis=0)
            print(i, extracted_label_data.shape)
        
        # 가속도 값, label 값, id_user값 추출
        segments = extracted_label_data[:, :, :3]
        extended_labels = extracted_label_data[:, :, 3]
        extended_id_user = extracted_label_data[:, :, 4]
        
        # 에러 방지용 print & 하나의 배치에 대해 하나의 레이블을 가져야 하는데, 현재는 하나의 타입스텝에 대해 레이블을 가지고 있다.
        # 따라서 2차원 데이터를 1차원 데이터로 압축해야 한다. 전체를 더하고 나누는 방법도 있다.(예, tf.reduce_mean(x, axis=1)) 
        # 하지만 이는 부동 소수점 오차 이슈의 가능성이 있기 때문에,for loop을 돌렸다.
        for label, user in zip(extended_labels, extended_id_user):
            if len(set(label)) != 1:
                print("한 타임스텝 내의 레이블에 다른게 포함됨")
                continue
                
            if len(set(user)) != 1:
                print("한 타임스텝 내의 user에 다른게 포함됨")
                continue
            
            labels.append(label[0])
            id_user.append(user[0])
        
        segments = np.asarray(segments, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        return segments, labels
    
def create_wisdm_2_0(filepath):
    dt = Dataset(filepath)
    df, num_classes = dt.read_data()
    
    x_data, y_data = dt.create_segments_and_labels(df)
    x_train = x_train / 20
    y_train = to_categorical(y_train, num_classes)
    
    print("x_train.shape : ", x_train.shape, "y_train.shape: ", y_train.shape)
    return x_train, y_train, num_classes, df