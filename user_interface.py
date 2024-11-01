import gradio as gr
import joblib
import numpy as np
from PIL import Image
import svm  # 调用自己创建的 svm 类
import time

def img2vector(img: np.array):
    img_normalization = np.round(img / 255)
    img_arr2 = np.reshape(img_normalization, (1, -1))
    return img_arr2

# 加载预训练的 SVM 模型
start_time = time.time()
model_path = "svm.model"  # 确保模型路径正确
clf = joblib.load(model_path)
print(time.time() - start_time)

# 图像处理和预测函数
def predict_digit(image):
    # 将图像转换为灰度图并调整到 28x28 像素的 MNIST 格式
    image = image.convert("L").resize((28, 28))
    data = np.array(image)
    data = img2vector(data)
    # 进行预测
    try:
        prediction = clf.predict(data)
        return f"预测的结果是: {prediction[0]}"
    except Exception as e:
        return f"预测时发生错误: {str(e)}"

# 自定义 CSS 样式
custom_css = """
#app-title {
    font-family: 'Arial', sans-serif;
    font-size: 5em;
    color: #FFFFFF;
    text-align: center;
    padding: 10px;
}

#app-description {
    text-align: center;
    color: #AAA;
    font-size: 1em;
    margin-bottom: 20px;
}

#upload-image {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 10px;
    width: 100%;
    height: 300px;
    margin: auto;
}

#result-output {
    font-size: 1.2em;
    font-weight: bold;
    color: #333;
    text-align: center;
    padding: 15px;
    border: 2px solid #4CAF50;
    border-radius: 10px;
    margin: 20px 0;
}

#button-container {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 10px;
}

.gr-button {
    font-size: 1em;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}

#submit-button {
    background-image: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
    color: white;
    border: none;
}

#submit-button:hover {
    background-image: linear-gradient(to top, #cfd9df 0%, #e2ebf0 100%);
}

#clear-button {
    background-image: linear-gradient(to top, #a18cd1 0%, #fbc2eb 100%);
    color: white;
    border: none;
}

#clear-button:hover {
    background-image: linear-gradient(to top, #cfd9df 0%, #e2ebf0 100%);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<div id='app-title'>手写数字识别</div>")
    gr.Markdown("<div id='app-description'>上传一个手写数字图像，我们的 SVM 模型将识别数字。请确保图像为灰度图且尺寸接近 28x28 像素。</div>")
    
    with gr.Column():
        upload_image = gr.Image(
            sources="upload",
            type="pil",
            label="上传手写体图像",
            elem_id="upload-image"
        )
        
        result_output = gr.Textbox(
            label="预测结果",
            placeholder="在这里查看预测结果",
            elem_id="result-output"
        )
    
    with gr.Row(elem_id="button-container"):
        submit_button = gr.Button("Submit", elem_id="submit-button")
        clear_button = gr.Button("Clear", elem_id="clear-button")

    submit_button.click(predict_digit, inputs=upload_image, outputs=result_output)
    clear_button.click(lambda: None, None, [upload_image, result_output], js="() => { return [null, '']; }")

# 运行 Gradio 界面，启用分享链接
demo.launch()



