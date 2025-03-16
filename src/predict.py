#这个文件的作用是：

# 加载 train.py 训练好的模型
# 处理输入文本
# 输出预测的 MBTI 类型

#运行方式
# cd D:\Workspace\AI_Workspace\MBTI_Transformer\src
# python predict.py


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dataset import load_data
from sklearn.preprocessing import LabelEncoder

# 加载标签编码器
df = load_data("../data/mbti_1.csv")
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["type"])

# 加载模型和分词器
model_path = "../models/mbti_bert"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_mbti(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()
    return label_encoder.inverse_transform([predicted_label])[0]

# 运行测试
if __name__ == "__main__":
    test_text = "I love coding and discussing deep learning!"
    prediction = predict_mbti(test_text)
    print(f"预测的 MBTI 类型: {prediction}")
