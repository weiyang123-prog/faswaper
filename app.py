from flask import Flask, request, jsonify, send_file, render_template, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import tempfile
from datetime import datetime
import json
import hashlib

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key_here'  # 用于session加密

# 用户数据存储文件
USERS_FILE = 'users.json'
FEEDBACK_FILE = 'feedback.json'

# 初始化用户数据文件
def init_files():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)

# 密码加密
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 初始化模型
base_path = os.getcwd()
face_app = FaceAnalysis(name='buffalo_l', root=base_path)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 加载换脸模型
swapper_model_path = os.path.join(base_path, 'models', "inswapper_128.onnx")
swapper = insightface.model_zoo.get_model(swapper_model_path, root=base_path)

def get_max_face(app, img):
    """获取图像中最大的人脸"""
    faces = app.get(img)
    if len(faces) < 1:
        return None

    areas = []
    for face in faces:
        bbox = face['bbox']
        area = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
        areas.append(area)
    index = np.argmax(areas)
    return faces[index]

@app.route('/')
def index():
    """首页"""
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)

        if username in users and users[username]['password'] == hash_password(password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='用户名或密码错误')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册页面"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('register.html', error='密码不一致')

        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)

        if username in users:
            return render_template('register.html', error='用户名已存在')

        users[username] = {
            'password': hash_password(password),
            'register_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

        session['username'] = username
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    """退出登录"""
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/generate', methods=['POST'])
def generate_image():
    if 'username' not in session:
        return jsonify({'error': '请先登录'}), 401

    try:
        # 获取上传的文件
        user_photo = request.files['userPhoto']
        costume_photo = request.files['costumePhoto']

        # 将文件转换为OpenCV图像格式
        user_array = np.frombuffer(user_photo.read(), np.uint8)
        costume_array = np.frombuffer(costume_photo.read(), np.uint8)

        img_user = cv2.imdecode(user_array, cv2.IMREAD_COLOR)
        img_costume = cv2.imdecode(costume_array, cv2.IMREAD_COLOR)

        if img_user is None or img_costume is None:
            return jsonify({'error': '无法读取图像文件'}), 400

        # 使用替换整个脸部的方式：服饰脸 -> 用户脸
        face_src = get_max_face(face_app, img_costume)
        face_tgt = get_max_face(face_app, img_user)
        base_image = img_user.copy()

        if face_tgt is None or face_src is None:
            return jsonify({'error': '未检测到人脸，请上传包含清晰人脸的图片'}), 400

        # 执行换脸
        result = swapper.get(base_image, face_tgt, face_src, paste_back=True)

        # 保存结果到临时文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{timestamp}.jpg"
        output_path = os.path.join('static', 'results', output_filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, result)

        # 返回结果URL
        result_url = f"/static/results/{output_filename}"

        return jsonify({
            'success': True,
            'imageUrl': result_url,
            'message': '图像生成成功'
        })

    except Exception as e:
        return jsonify({'error': f'处理过程中出现错误: {str(e)}'}), 500

@app.route('/static/results/<filename>')
def serve_result(filename):
    """提供生成的结果图像"""
    return send_file(os.path.join('static', 'results', filename))

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """提交反馈"""
    if 'username' not in session:
        return jsonify({'error': '请先登录'}), 401

    try:
        feedback_data = request.json
        feedback_data['username'] = session['username']
        feedback_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedbacks = json.load(f)

        feedbacks.append(feedback_data)

        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)

        return jsonify({'success': True, 'message': '反馈提交成功'})
    except Exception as e:
        return jsonify({'error': f'提交反馈失败: {str(e)}'}), 500

@app.route('/get_feedbacks')
def get_feedbacks():
    """获取反馈列表"""
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedbacks = json.load(f)
        return jsonify(feedbacks)
    except Exception as e:
        return jsonify({'error': f'获取反馈失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 初始化文件
    init_files()

    # 创建必要的目录
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True)
