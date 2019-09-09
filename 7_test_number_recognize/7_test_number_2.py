import numpy as np
from keras.models import load_model
import numpy as np
from PIL import Image

# 读取图片
def ImageToMatrix(filename):
    im = Image.open(filename)
    # change to greyimage
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='int')
    return data


# 读取模型
model = load_model('./model.h5')

# 读取数据并处理
data = ImageToMatrix('./6.png')
data = np.array(data)
data = data.reshape(1, 28, 28, 1)

# 预测并查看结果
result = model.predict_classes(data, batch_size=1, verbose=0)
print(result)
