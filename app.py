from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
import traceback

# Flask 앱 생성
app = Flask(__name__, static_folder='static')

# CORS 설정
CORS(app, resources={r"/*": {"origins": "*"}})

# MNIST 모델 로드
model = tf.keras.models.load_model('mnist_model.h5')

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.convert('L')  # 흑백 변환
    image = image.resize((28, 28), Image.Resampling.LANCZOS)  # Pillow 9.1.0 이하용, 최신은 Image.Resampling.LANCZOS 권장
    
    # 색 반전: canvas는 검정 글씨, MNIST는 흰색 글씨 기준
    img_array = np.array(image)
    img_array = 255 - img_array

    img_array = img_array / 255.0  # 정규화
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# 루트 경로 - HTML 파일 서빙
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# 예측 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        img_data = data['image']
        img_str = img_data.split(',')[1]  # 'data:image/png;base64,...' 에서 base64 부분만 추출
        img_bytes = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 디버깅용: 받은 원본 이미지 저장
        image.save('received.png')

        img_array = preprocess_image(image)
        pred = model.predict(img_array)
        digit = int(np.argmax(pred))
        confidence = float(np.max(pred))

        return jsonify({'digit': digit, 'confidence': confidence})

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)          # 터미널에 전체 에러 스택 출력
        app.logger.error(error_msg)  # Flask 로그에도 기록
        return jsonify({'error': str(e)}), 500

# 메인 실행
if __name__ == '__main__':
    app.run(debug=True)
