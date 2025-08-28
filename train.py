import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer,Qwen2_5_VLForConditionalGeneration
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os
from torch.utils.data import Dataset as TorchDataset

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/model")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/root/autodl-tmp/model", device_map="auto",
                                                        torch_dtype=torch.bfloat16, trust_remote_code=True, )
model.enable_input_require_grads() 

no_au = {"SN002": "/root/autodl-tmp/database/DISFA/SN002/5.jpg",
        "SN010": "/root/autodl-tmp/database/DISFA/SN010/270.jpg",
        "SN001": "/root/autodl-tmp/database/DISFA/SN001/450.jpg",
        "SN026": "/root/autodl-tmp/database/DISFA/SN026/5.jpg",
        "SN027": "/root/autodl-tmp/database/DISFA/SN027/10.jpg",
        "SN032": "/root/autodl-tmp/database/DISFA/SN032/2775.jpg",
        "SN030": "/root/autodl-tmp/database/DISFA/SN030/50.jpg",
        "SN009": "/root/autodl-tmp/database/DISFA/SN009/170.jpg",
        "SN016": "/root/autodl-tmp/database/DISFA/SN016/700.jpg",
        "SN003": "/root/autodl-tmp/database/DISFA/SN003/1.jpg",
        "SN029": "/root/autodl-tmp/database/DISFA/SN029/1.jpg",
        "SN023": "/root/autodl-tmp/database/DISFA/SN023/50.jpg",
        "SN025": "/root/autodl-tmp/database/DISFA/SN025/100.jpg",
        "SN008": "/root/autodl-tmp/database/DISFA/SN008/50.jpg",
        "SN005": "/root/autodl-tmp/database/DISFA/SN005/10.jpg",
        "SN007": "/root/autodl-tmp/database/DISFA/SN007/400.jpg",
        "SN017": "/root/autodl-tmp/database/DISFA/SN017/1.jpg",
        "SN013": "/root/autodl-tmp/database/DISFA/SN013/1.jpg",
        "SN018": "/root/autodl-tmp/database/DISFA/SN018/1.jpg",
        "SN011": "/root/autodl-tmp/database/DISFA/SN011/30.jpg",
        "SN028": "/root/autodl-tmp/database/DISFA/SN028/1.jpg",
        "SN012": "/root/autodl-tmp/database/DISFA/SN012/900.jpg",
        "SN006": "/root/autodl-tmp/database/DISFA/SN006/1.jpg",
        "SN031": "/root/autodl-tmp/database/DISFA/SN031/60.jpg",
        "SN021": "/root/autodl-tmp/database/DISFA/SN021/0.jpg",
        "SN024": "/root/autodl-tmp/database/DISFA/SN024/50.jpg",
        "SN004": "/root/autodl-tmp/database/DISFA/SN004/130.jpg"}


def process_func(example):
    """
    将数据集进行预处理
    """
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  
    no_au_file_path = no_au[file_path.split("/")[-2]]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{no_au_file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "The appearance changes of AU1(Inner Brow Raiser):Pulls the inner portion of the eyebrows upwards.\nThe appearance changes of AU2(Outer Brow Raiser):pulling the eyebrows and the adjacent skin in the lateral portion of the forehead upwards towards the hairline.Pulls the lateral (outer) portion of the eyebrows upwards.\nThe appearance changes of AU4(Brow Lowerer):Lowers the eyebrow. In different instances it may be only the inner portion of the eyebrow that is lowered or it may be both inner and central portions that are lowered, or it may appear that the entire eyebrow is lowered.Or,Brows pulled together, which can see a wrinkle or muscle bulge between brows.Pushes the eye cover fold downwards and may narrow the eye aperture.\nThe appearance changes of AU6(Cheek Raiser):Draws skin towards the eye from the temple and cheeks as the outer band of muscle around the eye constricts.Raises the infraorbital triangle, lifting the cheek upwards.Pushes the skin surrounding the eye towards the eye socket, which can narrow the eye aperture, bag or wrinkle the skin below the eye, and push the eye cover fold down and/or change its shape.\nThe appearance changes of AU9(Nose Wrinkler):Pulls the skin along the sides of the nose upwards.Pulls the infraorbital triangle upwards, causing the infraorbital furrow to wrinkle.Lowers the medial portion of the eyebrows, which tends to conceal any raising of the inner corners of the brow if AU 1 were to act.Narrows the eye aperture due to the actions described in appearance changes.\nAU12(Lip Corner Puller):This AU pulls the corners of the lips back and upward,which is a common component of a smile.\nAU25(Lips Part):The lips part, which may expose the inner mucosal area of the lips more.\nThe appearance changes of AU26(Jaw Drop):Marked and unambiguous dropping of the mandible by relaxation.If the lips part, space between the teeth may be seen; score 25+26.Mouth appears as if jaw has dropped or fallen with no sign of the jaw being pulled open or stretching of the lips due to opening the jaw wide.\nThe two pictures depict the face of the same person. Only consider AU1, AU2, AU4, AU6, AU9, AU12, AU25, and AU26. In the first picture, there are no Action Units (AUs) present on this person's face. By comparing the two pictures, find out which Action Units (AUs) occur on this person's face in the second picture.The evidence of the occurrence of AU is not obvious, or the occurrence of AU is judged even if there is the slightest indication.Try to determine that AU has occurred as much as possible.Even if it's uncertain whether AU has occurred or not, it should be regarded as having occurredm,specially AU1 and AU2."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  

    image_inputs, video_inputs = process_vision_info(messages)  
    print(type(image_inputs))
    for i, img in enumerate(image_inputs):
       print(f"Image {i}: type={type(img)}, size={getattr(img, 'size', None)}")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"

    )
    inputs = {key: value.tolist() for key, value in inputs.items()} 

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(inputs["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw'])  

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs['pixel_values'],
        "image_grid_thw": inputs['image_grid_thw']
    }



    return data


class ProcessedDataset(TorchDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(data_dir, "r") as f:
            all_data = json.load(f)
        self.data = all_data[::5]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = process_func(data)
        return data


train_dataset = ProcessedDataset("/root/autodl-tmp/code/train_and_contrast.json")


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","attn.proj","attn.qkv"],
    inference_mode=False,  
    r=128,  
    lora_alpha=64,  
    lora_dropout=0.1,  
    bias="none",
)


peft_model = get_peft_model(model, config)
# peft_model = PeftModel.from_pretrained(model, model_id="/home/qwen/output/Qwen2.5-VL-3B/checkpoint-3633", config=config)


args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/2VL-7B",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=1,
    save_strategy="epoch",
    # save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)


swanlab_callback = SwanLabCallback(
    project="QwenVLand",
    experiment_name="qwen2-vl-2B-contrast",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "/root/autodl-fs/promt/train_and_contrast.json",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "miao: ",
        "train_data_number": len(train_dataset),
        "lora_rank": 128,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "none",
    },
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


trainer.train()
swanlab.finish()
