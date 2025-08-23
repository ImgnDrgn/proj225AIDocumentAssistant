import sys
from datasets import load_dataset, Dataset
import itertools
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments
import evaluate
import numpy as np

# import torch
import os

# 禁用符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 解析命令行参数
if len(sys.argv) != 3:
    print("正确用法：python finetune.py <模型名称> <模型保存路径>")
    print("示例：python finetune.py FlagAlpha/Llama2-Chinese-13b-Chat ./FlagAlpha13b")
    sys.exit(1)

MODEL_NAME = sys.argv[1]  # 传入的预训练模型名
OUTPUT_DIR = sys.argv[2]  # 传入的模型保存路径


# ----------------------------------- 加载数据集 ---------------------------------- #
# 加载原始数据
raw_data = load_dataset("json", data_files="DeepinWiki.json", split="train")


# 展平+格式转换函数
def process_data(example):
    flat_samples = []
    # 提取title
    title = example["data"].get("title", "Unknown")
    for para in example["data"]["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            # 直接重组answers格式（无需筛选，数据均有效）
            target_answers = {
                "answer_start": [qa["answers"][0]["answer_start"]],  # 整型→单元素列表
                "text": [qa["answers"][0]["text"]],  # 字符串→单元素列表
            }
            flat_samples.append(
                {
                    "context": context,
                    "question": qa["question"],
                    "answers": target_answers,  # 列表->字典
                    "id": qa["id"],
                    "title": title,
                }
            )
    return {"samples": flat_samples}


# 数据处理+展平
mapped_data = raw_data.map(process_data)
flattened_samples = list(itertools.chain(*mapped_data["samples"]))

# 重建Dataset
flattened_data = Dataset.from_dict(
    {
        "context": [s["context"] for s in flattened_samples],
        "question": [s["question"] for s in flattened_samples],
        "answers": [s["answers"] for s in flattened_samples],
        "id": [s["id"] for s in flattened_samples],
        "title": [s["title"] for s in flattened_samples],
    }
)

# 拆分为训练集和测试集，测试集占0.2
deepin_dataset = flattened_data.train_test_split(test_size=0.2)

# print(squad["train"][0])  # 验证结果


# ------------------------------------ 预处理 ----------------------------------- #
# 使用模型配套的分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# 预处理函数
def preprocess_function(examples):
    questions = [q.strip() if isinstance(q, str) else "" for q in examples["question"]]
    contexts = [c.strip() if isinstance(c, str) else "" for c in examples["context"]]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        add_special_tokens=True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs["overflow_to_sample_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i in range(len(offset_mapping)):
        sample_idx = sample_map[i]
        current_answer = answers[sample_idx]

        start_char = current_answer["answer_start"][0]
        end_char = start_char + len(current_answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)
        if sequence_ids is None:
            start_positions.append(-100)
            end_positions.append(-100)
            continue

        context_start = next(
            (idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1), None
        )
        # 反向找最后一个属于context的token（seq_id=1）
        context_end = next(
            (
                idx
                for idx, seq_id in reversed(list(enumerate(sequence_ids)))
                if seq_id == 1
            ),
            None,
        )

        if context_start is None or context_end is None:
            start_positions.append(-100)
            end_positions.append(-100)
            continue

        if (
            offset_mapping[i][context_end][1] < start_char
            or offset_mapping[i][context_start][0] > end_char
        ):
            start_positions.append(-100)
            end_positions.append(-100)
        else:
            start_token = context_start
            while (
                start_token <= context_end
                and offset_mapping[i][start_token][0] <= start_char
            ):
                start_token += 1
            start_positions.append(start_token - 1)

            end_token = context_end
            while (
                end_token >= context_start
                and offset_mapping[i][end_token][1] >= end_char
            ):
                end_token -= 1
            end_positions.append(end_token + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


# 应用预处理（批量处理+移除文本列）
tokenized_deepin = deepin_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=deepin_dataset["train"].column_names,
)

# 生成训练批次数据
data_collator = DefaultDataCollator()

# ------------------------------------ 训练 ------------------------------------ #
# 加载模型
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, device_map="auto")
# 加载模型后打印状态（确认是否成功）
print("模型加载成功！")
print(f"模型设备：{next(model.parameters()).device}")  # 应输出 cuda:0（GPU）

# 配置TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # 模型保存路径
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,  # 不需要推送到hf的hub
    save_strategy="epoch",
    load_best_model_at_end=True,  # 训练结束后加载最优模型
    metric_for_best_model="loss",  # 以损失值作为最优判断标准
    greater_is_better=False,  # 损失值越小越好
    save_total_limit=3,  # 限制保留的最多3个快照
    logging_steps=10,  # 每10步记录一次日志
    logging_dir="./logs",  # 日志目录
    report_to="none",  # 禁用所有集成报告器
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_deepin["train"],
    eval_dataset=tokenized_deepin["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

# 启动训练
trainer.train()

# 保存训练好的模型
best_model_path = f"{OUTPUT_DIR}/best_model"
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)
print(f"最优模型已保存到：{best_model_path}")
