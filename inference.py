"""
多轮视觉思考链推理脚本 (Visual Thinking Chain Inference)

原理：
1. 模型生成思考过程
2. 检查是否完成（包含 </think> 且之后有非空的 <answer>...</answer>）
3. 若未完成，将生成文本渲染为图片作为"视觉记忆"
4. 累积图片输入，继续生成
5. 重复直到完成或达到最大轮数
"""

import os
import time
import torch
from typing import List, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ============ 配置区域 ============
MODEL_PATH = "/home/wyb/ckpt/vtc-r1-glyph"
CONFIG_PATH = "./evaluation/config_en.json"
OUTPUT_DIR = "./output_images"
MAX_EPOCHS = 8
MAX_NEW_TOKENS = 8192

SYSTEM_PROMPT = (
    "These images record your previous reasoning process. "
    "Based on this reasoning, continue and complete the final answer. "
    "Do not restart the reasoning.\n"
    "If no images are provided, start the reasoning from scratch."
)


# ============ 核心函数 ============

def check_completion(text: str) -> bool:
    """
    检查生成是否结束。
    条件：
    1. 必须包含 </think>
    2. </think> 之后必须包含 <answer> 和 </answer>
    3. <answer> 和 </answer> 之间的内容非空
    """
    if '</think>' not in text:
        return False
    
    # 获取 </think> 之后的内容
    content_after_think = text.split('</think>')[-1]
    
    # 检查是否存在成对的 answer 标签
    if '<answer>' in content_after_think and '</answer>' in content_after_think:
        answer_content = content_after_think.split('<answer>')[1].split('</answer>')[0]
        return bool(answer_content.strip())
    
    return False


def render_text_to_images(text: str, unique_id: str) -> List[Image.Image]:
    """
    将生成的思考文本渲染为图片列表（作为视觉记忆）。
    """
    # 延迟导入，避免不需要时的依赖
    from evaluation.word2png_function import text_to_images
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    paths = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        unique_id=f"{unique_id}_{int(time.time())}_{hash(text) % 100000}"
    )
    
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"Warning: Failed to load image {p}: {e}")
    
    return images


def build_messages(question: str, images: List[Image.Image]) -> List[dict]:
    """
    构建多模态消息列表。
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": question}] +
                [{"type": "image", "image": img} for img in images]
            )
        }
    ]


def generate_once(
    model, 
    processor, 
    messages: List[dict], 
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """
    单轮生成。
    """
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 只解码新生成的部分
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=False
    )
    
    return output_text


def multi_round_inference(
    question: str,
    initial_images: Optional[List[Image.Image]] = None,
    model_path: str = MODEL_PATH,
    max_epochs: int = MAX_EPOCHS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    verbose: bool = True
) -> dict:
    """
    多轮视觉思考链推理。
    
    Args:
        question: 用户问题
        initial_images: 初始图片列表（默认为空）
        model_path: 模型路径
        max_epochs: 最大推理轮数
        max_new_tokens: 每轮最大生成 token 数
        verbose: 是否打印过程信息
    
    Returns:
        dict: {
            "final_output": 最终输出文本,
            "all_outputs": 所有轮次的输出列表,
            "num_rounds": 总轮数,
            "is_complete": 是否正常完成
        }
    """
    # 加载模型和处理器
    if verbose:
        print(f"Loading model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 初始化状态
    accumulated_images = list(initial_images) if initial_images else []
    all_outputs = []
    is_complete = False
    final_output = ""
    
    for epoch in range(max_epochs):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{max_epochs}")
            print(f"Accumulated images: {len(accumulated_images)}")
        
        # 构建消息
        messages = build_messages(question, accumulated_images)
        
        # 生成
        output_text = generate_once(model, processor, messages, max_new_tokens)
        all_outputs.append(output_text)
        
        
        # 检查是否完成
        if check_completion(output_text):
            if verbose:
                print("✓ Generation complete!")
            final_output = output_text
            is_complete = True
            break
        
        # 未完成：渲染为图片并累积
        if verbose:
            print("→ Not complete, rendering to images...")
        
        new_images = render_text_to_images(output_text, f"round_{epoch}")
        
        if not new_images:
            if verbose:
                print("⚠ No images rendered, stopping.")
            final_output = output_text
            break
        
        accumulated_images.extend(new_images)
        if verbose:
            print(f"  Added {len(new_images)} images")
    
    # 若达到最大轮数仍未完成
    if not final_output:
        final_output = all_outputs[-1] if all_outputs else ""
        if verbose:
            print(f"\n⚠ Reached max epochs ({max_epochs}), returning last output.")
    
    return {
        "final_output": final_output,
        "all_outputs": all_outputs,
        "num_rounds": len(all_outputs),
        "is_complete": is_complete
    }


# ============ 示例使用 ============

if __name__ == "__main__":
    # 示例问题
    question = "Solve for $x: 3^{2x} + 19 = 10^x$."   # gt: 2
    
    
    # 执行多轮推理
    result = multi_round_inference(
        question=question,
        initial_images=[],
        max_epochs=MAX_EPOCHS,
        verbose=True
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Total rounds: {result['num_rounds']}")
    print(f"Is complete: {result['is_complete']}")
    print(f"\nFinal output:\n{result['final_output']}")
