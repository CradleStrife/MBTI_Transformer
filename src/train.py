#这个文件的作用是：

#加载 dataset.py 处理后的数据
#使用 BERT 进行 MBTI 文本分类
#训练模型并保存

#运行方式
# cd D:\Workspace\AI_Workspace\MBTI_Transformer\src
# python train.py

import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from dataset import load_data, MBTIDataset
from sklearn.preprocessing import LabelEncoder

# 读取数据
df = load_data("../data/mbti_1.csv")

# 进行标签编码
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["type"])

# 生成 PyTorch 数据集
dataset = MBTIDataset(df["posts"].tolist(), df["label"].values)

# 加载 BERT 预训练模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=16)

# 训练参数
training_args = TrainingArguments(
    output_dir="../models",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="../logs",
)

# 使用 Hugging Face `Trainer` 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("../models/mbti_bert")
print("✅ 模型训练完成，已保存到 models/mbti_bert")
