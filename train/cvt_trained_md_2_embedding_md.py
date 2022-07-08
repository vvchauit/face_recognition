from keras.models import load_model
from keras.models import Model


# summary: chuyển model classifier đã huấn luyện thành model trích xuất đặc trưng
# params:
# 	init
# 		base_model_path: địa chỉ model classifier đã huấn luyện
# 		embedding_model_weights_path: địa chỉ lưu trọng số model trích xuất đặc trưng
# 	return
# 		model: model trích xuất đặc trưng
def cvt_trained_md_2_embedding_md(base_model_path, embedding_model_weights_path):
    base_model = load_model(base_model_path)
    model = Model(base_model.inputs, base_model.layers[-2].output)
    model.save_weights(embedding_model_weights_path)
    return model
