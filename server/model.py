from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
from flask_cors import CORS

from segment_anything import SamPredictor, sam_model_registry

app = Flask(__name__)
CORS(app)


def init():
    checkpoint = "/Users/zaihui-101/dev/model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    return predictor


@app.route('/segment_everything_box_model', methods=['POST'])
def process_image():
    # 从请求中获取图片数据
    image_data = request.data

    # 将二进制数据转换为PIL图像对象
    pil_image = Image.open(io.BytesIO(image_data))

    # 将图像转换为Numpy数组
    np_image = np.array(pil_image)

    predictor = init()
    predictor.set_image(np_image)

    image_embedding = predictor.get_image_embedding().cpu().numpy()

    # 将二进制数据转换为Base64字符串
    result_base64 = base64.b64encode(image_embedding.tobytes()).decode('utf-8')

    # 添加到结果列表
    result_list = [result_base64]
    return jsonify(result_list)


if __name__ == '__main__':
    app.run()
