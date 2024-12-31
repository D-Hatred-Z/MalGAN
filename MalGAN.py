import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Hàm ánh xạ nhãn
def map_labels(df):
    labels = df['label'].copy()
    label_map = {'Benign': 0}
    label_map.update({label: 1 for label in labels.unique() if label != 'Benign'})
    df['label'] = df['label'].map(label_map)
    return df

# Hàm tiền xử lý dữ liệu
def preprocess_malware_data(df):
    df = df.drop(columns=['Unnamed: 0', 'local_orig', 'local_resp', 'uid'])
    df = map_labels(df)
    
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    
    X = df.drop(columns=['label']).values
    y = df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = to_categorical(y)

    return X, y

# Hàm tải dữ liệu
def load_malware_dataset():
    df = pd.read_csv('/kaggle/input/malware/iot23_combined.csv')
    X, y = preprocess_malware_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Hàm xây dựng Generator
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    return model

# Hàm xây dựng Discriminator
def build_discriminator(input_dim, num_classes):
    model = tf.keras.Sequential([
        layers.Dense(512, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

# Hàm huấn luyện Black-Box Model (Voting Classifier)
def train_black_box_model(X_train, y_train):
    logistic_model = LogisticRegression(random_state=42)
    dctree_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    voting_model = VotingClassifier(estimators=[
        ('logistic', logistic_model), 
        ('dctree', dctree_model), 
        ('rf', rf_model), 
        ('xgb', xgb_model)
    ], voting='soft')
    
    voting_model.fit(X_train, np.argmax(y_train, axis=1))
    return voting_model

# Hàm Black-Box Detector
def black_box_detector(model, adversarial_data):
    return np.argmax(model.predict_proba(adversarial_data), axis=1)

# Hàm huấn luyện MalGAN
def train_malgan(generator, discriminator, black_box_model, X_malware, y_malware, X_benign, y_benign, epochs=1000, batch_size=256, num_classes=2):
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    generator_losses, discriminator_losses = [], []
    
    for epoch in range(epochs):
        # Lấy một batch từ dữ liệu mã độc
        idx_malware = np.random.randint(0, X_malware.shape[0], batch_size)
        real_data = X_malware[idx_malware]
        real_labels = y_malware[idx_malware]
        
        # Sinh dữ liệu giả và thêm noise
        noise = np.random.normal(0, 1, (batch_size, real_data.shape[1]))
        fake_data = generator.predict(noise)
        fake_data += np.random.normal(0, 0.1, fake_data.shape)  # Thêm noise vào dữ liệu giả
        
        # Kết hợp với dữ liệu benign
        idx_benign = np.random.randint(0, X_benign.shape[0], batch_size)
        benign_data = X_benign[idx_benign]
        combined_fake_data = np.concatenate([fake_data, benign_data], axis=0)
        
        # Gán nhãn cho dữ liệu giả qua Black-Box Detector
        fake_labels = black_box_detector(black_box_model, combined_fake_data)
        fake_labels = to_categorical(fake_labels, num_classes=num_classes)
        
        # Kết hợp dữ liệu thật và dữ liệu giả đã được gán nhãn
        combined_data = np.concatenate([real_data, combined_fake_data], axis=0)
        combined_labels = np.concatenate([real_labels, fake_labels], axis=0)
        
        # Huấn luyện Discriminator
        with tf.GradientTape() as tape:
            d_real_logits = discriminator(real_data, training=True)
            d_fake_logits = discriminator(combined_fake_data, training=True)
            d_real_loss = binary_crossentropy(real_labels, d_real_logits)
            d_fake_loss = binary_crossentropy(fake_labels, d_fake_logits)
            d_loss = d_real_loss + d_fake_loss
        
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        
        # Huấn luyện Generator
        with tf.GradientTape() as tape:
            fake_data = generator(noise, training=True)
            g_loss = binary_crossentropy(real_labels, discriminator(fake_data, training=True))
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
        
        generator_losses.append(g_loss.numpy())
        discriminator_losses.append(d_loss.numpy())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss.numpy():.4f}, G Loss: {g_loss.numpy():.4f}")
    
    return generator_losses, discriminator_losses

# Đọc dữ liệu và chia thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = load_malware_dataset()

# Tách dữ liệu benign và malware
X_benign = X_train[np.argmax(y_train, axis=1) == 0]
y_benign = y_train[np.argmax(y_train, axis=1) == 0]
X_malware = X_train[np.argmax(y_train, axis=1) != 0]
y_malware = y_train[np.argmax(y_train, axis=1) != 0]

# Xây dựng các mô hình Generator và Discriminator
input_dim = X_train.shape[1]
generator = build_generator(input_dim, input_dim)
discriminator = build_discriminator(input_dim, 2)

# Huấn luyện Black-Box Model (Voting Classifier)
black_box_model = train_black_box_model(X_train, y_train)

# Huấn luyện MalGAN với toàn bộ dữ liệu và Black-Box Detector
g_losses, d_losses = train_malgan(generator, discriminator, black_box_model, X_malware, y_malware, X_benign, y_benign, epochs=1000)

# Vẽ đồ thị Loss
def plot_losses(generator_losses, discriminator_losses):
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Losses During Training")
    plt.show()

plot_losses(g_losses, d_losses)

-------------------------------------------------------------------------------------------------

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf

# Tạo dữ liệu giả (adversarial examples)
noise = np.random.normal(0, 1, (X_test.shape[0], X_test.shape[1]))  # Tạo nhiễu ngẫu nhiên cho generator
X_adv = generator.predict(noise)  # Sinh adversarial examples từ generator

# Khởi tạo các mô hình thành phần
logistic_model = LogisticRegression(random_state=42)
dctree_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

models = {
    'Logistic Regression': logistic_model,
    'Decision Tree': dctree_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# Chuyển nhãn one-hot thành nhãn nguyên
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Đánh giá từng mô hình
for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    
    # Huấn luyện mô hình trên dữ liệu thật
    model.fit(X_train, y_train_labels)

    # Dự đoán trên dữ liệu thật
    real_preds = model.predict(X_test)
    
    # Gán nhãn cho adversarial examples bằng mô hình Black-Box
    adv_preds = black_box_detector(model, X_adv)

    # Báo cáo chi tiết cho dữ liệu thật
    print("Real Data Classification Report:")
    print(classification_report(y_test_labels, real_preds))

    # Khi đánh giá adversarial data, nhãn của chúng sẽ là 1 (hoặc bất kỳ nhãn nào bạn muốn)
    adv_labels = np.ones_like(adv_preds)  # Tạo nhãn là 1 cho adversarial examples

    # Báo cáo chi tiết cho dữ liệu giả
    print("Adversarial Data Classification Report:")
    print(classification_report(adv_labels, adv_preds))

    # Kết hợp dữ liệu kiểm tra và dữ liệu giả
    combined_data = np.concatenate((X_test, X_adv), axis=0)
    combined_labels = np.concatenate((y_test_labels, adv_labels), axis=0)

    # Dự đoán trên dữ liệu kết hợp
    combined_preds = model.predict(combined_data)

    # Báo cáo chi tiết cho dữ liệu kết hợp
    print("Combined Data Classification Report:")
    print(classification_report(combined_labels, combined_preds))
