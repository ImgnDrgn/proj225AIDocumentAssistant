# 导入必要的库
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer, default_data_collator
from datasets import load_dataset
import numpy as np
import torch
import os

# 设置环境
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 在代码开头添加选择逻辑
model_path = "distilbert/distilbert-base-uncased"  # 默认用原始模型
if os.path.exists("./qa_model/config.json"):  # 检查是否有已保存的模型
    answer = input("检测到已保存的模型，是否加载？(y/n): ")
    if answer.lower() == "y":
        model_path = "./qa_model"

# 1. 加载模型和tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
#使用之前模型
model_path = "./qa_model" if os.path.exists("./qa_model/config.json") else "distilbert/distilbert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# 2. 加载并预处理数据集（使用更小的子集加速训练）
squad = load_dataset("squad")

# 简化版预处理函数
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=256,  # 减少长度加速处理
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    # 处理答案位置
    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")
    
    tokenized_inputs["start_positions"] = []
    tokenized_inputs["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            tokenized_inputs["start_positions"].append(cls_index)
            tokenized_inputs["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized_inputs.sequence_ids(i)
            
            # 找到答案的token范围
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
                
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
                
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_inputs["start_positions"].append(cls_index)
                tokenized_inputs["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_inputs["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_inputs["end_positions"].append(token_end_index + 1)
    
    return tokenized_inputs

# 使用更小的数据集
train_dataset = squad["train"].select(range(2000))  # 进一步减少到500个样本
eval_dataset = squad["validation"].select(range(100))  # 100个验证样本

# 预处理数据集
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./qa_model",
    eval_steps=100,  # 每100步评估一次
    per_device_train_batch_size=8 if device == "cuda" else 4,
    per_device_eval_batch_size=8 if device == "cuda" else 4,
    learning_rate=3e-5,
    weight_decay=0.01,
    num_train_epochs=2,  # 减少训练轮次
    save_total_limit=1,
    logging_steps=20,
    disable_tqdm=False  # 显示进度条
)

# 4. 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=default_data_collator,
    tokenizer=tokenizer
)

# 5. 开始训练
print("开始训练...")
trainer.train()

# 6. 评估
eval_results = trainer.evaluate()
print("评估结果:", eval_results)

# 7. 保存模型（用于后续测试）
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")

# 8. 测试单样本预测
print("\n===== 测试单样本预测 =====")
sample = eval_dataset[0]
inputs = tokenizer(
    sample["question"],
    sample["context"],
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)  # 确保输入在正确设备上

with torch.no_grad():
    outputs = model(**inputs)

answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"问题: {sample['question']}")
print(f"上下文片段: {sample['context'][:100]}...")
print(f"\n真实答案: {sample['answers']['text'][0]}")
print(f"预测答案: {answer}")
print(f"答案位置: [{answer_start.item()}, {answer_end.item()}]")

