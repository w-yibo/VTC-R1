import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

from typing import List, Dict, Any
from PIL import Image
from word2png_function import text_to_images
from transformers import AutoProcessor
CONFIG_EN_PATH = "./config_en.json"
OUTPUT_DIR = "./output_for_count_qwen"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--use_glyph", action="store_true")
    parser.add_argument("--max_epoch", default=10, type=int)
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)

    # shuffle
    if args.shuffle:
        random.shuffle(examples, seed=datetime.now().timestamp())

    # select start and end
    examples = examples[args.start:len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'
    # out_file = f'{args.output_dir}/{model_name}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    out_file = f'{args.output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl'
    os.makedirs(f'{args.output_dir}/{data_name}', exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [f for f in os.listdir(f"{args.output_dir}/{data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]    
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    # print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if args.use_vllm:
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus), trust_remote_code=True, mm_processor_cache_gb=20)
        tokenizer = None
    else:
        llm, tokenizer =  load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

    # infer & eval
    data_list = args.data_names.split(',')
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))
    
    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append({
        "acc": sum([result["acc"] for result in results]) / len(results),
    })
    
    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
def thought_check_completion_status(completion: str) -> bool:
    """
    检查生成是否结束。
    逻辑：
    1. 必须包含 </think>
    2. </think> 之后的部分必须包含 <answer> 和 </answer>
    3. <answer> 和 </answer> 之间的内容非空
    """
    if '</think>' not in completion:
        return False
    
    parts = completion.split('</think>')
    if len(parts) > 1: 
        # 获取 </think> 之后的内容
        content_after_think = parts[-1]
        
        # 检查是否存在成对的标签
        if '<answer>' in content_after_think and '</answer>' in content_after_think:
            # 提取标签中间的内容
            # 这里默认取第一个 <answer> 和其后的第一个 </answer> 之间的内容
            answer_content = content_after_think.split('<answer>')[1].split('</answer>')[0]
            
            # 检查内容是否非空
            if answer_content.strip():
                return True
                
    return False

def render_text_to_pil_imgs(text: str, data_name, idx) -> List[Image.Image]:
    """
    将生成的思考文本渲染为图片列表。
    """
    # 这里假设 text_to_images 是你现有的函数
    # text = text.replace("Got it, let's continue.", "")
    # text = re.sub(r'</?think>', '', text)
    paths = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR + f"/{data_name}",
        config_path=CONFIG_EN_PATH,
        unique_id=f"{idx}_{int(time.time())}_{hash(text)}" # 增加 hash 避免文件名冲突
    )
    pil_imgs = []
    for p in paths:
        try:
            pil_imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return pil_imgs

def build_single_prompt(user_content: str, image_count: int, processor) -> str:
    """
    构建单个样本的 Prompt 字符串。
    每次由于 image 数量变化，都需要重新构建。
    """
    SYSTEM_PROMPT = (
        "These images record your previous reasoning process. "
        "Based on this reasoning, continue and complete the final answer. "
        "Do not restart the reasoning.\n"
        "If no images are provided, start the reasoning from scratch."
    )
    
    # 动态生成 image placeholder
    # img_prefix = "<image>" * image_count
    placeholders = [{"type": "image", "image": "img"} for _ in range(image_count)]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [*placeholders, {"type": "text", "text": user_content}]}
    ]
    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {"role": "user", "content": user_content}
    # ] 
    
    # 生成最终的 prompt 字符串
    prompt_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(prompt_str)
    # exit()
    return prompt_str

class RequestState:
    """用于跟踪每个样本的状态"""
    def __init__(self, original_idx, user_content, initial_images):
        self.original_idx = original_idx
        self.user_content = user_content # 原始问题保持不变
        self.accumulated_images = list(initial_images) # 历史图片 + 新渲染的思考图片
        self.is_finished = False
        self.final_completion = None
        self.all_generated_text = []
        self.last_generated_text = ""

def batch_vllm_glyph(input_data, llm, args, stop_words, model_name, data_name):
    """
    input_data: 列表，每个元素格式建议为 (id, user_content, [initial_images]) 
                或者根据你之前的 inputs 调整
    """
    processor = AutoProcessor.from_pretrained(model_name)
    # processor = llm.get_tokenizer()
    MAX_EPOCHS = args.max_epoch
    
    # 1. 初始化状态列表
    states = []
    for i, item in enumerate(input_data):
        # 假设 input_data 里的 item 结构是 [id, text, images]
        # 如果只是 text list，请自行调整
        user_content = item[1]
        initial_images = item[2] if len(item) > 2 else []
        states.append(RequestState(i, user_content, initial_images))

    # 最终结果容器
    final_outputs = [None] * len(states)
    all_generated_text = [[] for _ in range(len(states))]

    for epoch in range(MAX_EPOCHS):
        # 2. 筛选出还需要继续生成的样本
        active_states = [s for s in states if not s.is_finished]
        
        if not active_states:
            print("All requests finished.")
            break
            
        print(f"Epoch {epoch}: Processing {len(active_states)} active requests...")

        # 3. 准备 vLLM 的输入 Batch
        prompts = []
        multi_modal_datas = []
        
        for state in active_states:
            # 重新构建 prompt，包含当前所有累积的图片
            prompt_str = build_single_prompt(
                state.user_content, 
                len(state.accumulated_images), 
                processor
            )
            prompts.append(prompt_str)
            
            # 准备多模态数据
            if state.accumulated_images:
                multi_modal_datas.append({"image": state.accumulated_images})
            else:
                # 如果没有图片（第一轮），vLLM 可能需要特殊处理或传空字典，视版本而定
                # 大多数 vLLM 版本如果不含 <image> token 且不传 multi_modal_data 即可
                multi_modal_datas.append(None) 

        # 4. 调用 vLLM 生成
        # 构建 inputs 列表，适配 vLLM 0.6+ 推荐方式，或者使用 inputs 参数
        # 这里使用比较通用的 prompt + multi_modal_data 分离方式
        # 注意：vLLM 的 generate 如果传入 list，需要一一对应
        
        # 构造 inputs 列表给 generate
        inputs_batch = []
        for p, mm in zip(prompts, multi_modal_datas):
            item = {"prompt": p}
            if mm:
                item["multi_modal_data"] = mm
            inputs_batch.append(item)

        batch_outputs = llm.generate(
            inputs_batch,
            sampling_params=SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                skip_special_tokens=False,
            )
        )
        batch_outputs = sorted(batch_outputs, key=lambda x: int(x.request_id))
        # 5. 处理输出并更新状态
        # vLLM 返回的 batch_outputs 顺序与输入的 inputs_batch 一致
        for i, output in enumerate(batch_outputs):
            state = active_states[i] # 找回对应的状态对象
            generated_text = output.outputs[0].text

            state.last_generated_text = generated_text
            state.all_generated_text.append(generated_text)
            # 检查是否结束
            if thought_check_completion_status(generated_text):
                state.is_finished = True
                state.final_completion = generated_text
                # 将结果存入最终列表
                final_outputs[state.original_idx] = generated_text
                all_generated_text[state.original_idx] = state.all_generated_text
            else:
                # 没结束：渲染生成的“思考过程”为图片
                # 注意：这里我们**不**保留文本，而是将文本转化为图片，作为下一轮的“记忆”
                new_imgs = render_text_to_pil_imgs(generated_text, data_name, state.original_idx)
                
                if not new_imgs:
                    # 如果渲染失败或没有生成图片，可能需要强制停止或保留文本
                    # 这里为了防止死循环，如果渲染不出东西，视为结束或出错
                    print(f"Warning: No images rendered for ID {state.original_idx}, stopping.")
                    state.is_finished = True
                    final_outputs[state.original_idx] = generated_text
                else:
                    state.accumulated_images.extend(new_imgs)
                    # 下一轮循环时，build_single_prompt 会使用新的 accumulated_images 长度

    # 6. 处理达到最大轮数仍未结束的样本
    for state in states:
        if final_outputs[state.original_idx] is None:
            # 超时，强制返回最后一次的生成结果（可能是中间过程）
            print(f"ID {state.original_idx} reached max epochs. Returning last output.")
            final_outputs[state.original_idx] = state.last_generated_text
            all_generated_text[state.original_idx] = state.all_generated_text
    return final_outputs, all_generated_text

def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)


    # repeat n times
    input_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 4

    # stop words TODO: make it more general
    stop_words = ["</s>"]

    if args.prompt_type in ['cot']:
        stop_words.extend(["\n\nQuestion:", "\n\nProblem:"])
    if args.prompt_type in ['pal', 'tool-integrated', 'tora']:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_words.extend(["Instruction", "Response"])
    print("Stop words:", stop_words)

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            if not args.use_glyph:
                outputs = llm.generate(prompts, SamplingParams(
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_tokens_per_call,
                                n=1,
                                stop=stop_words,
                ))
                outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
                outputs = [output.outputs[0].text for output in outputs]
            else:
                glyph_prompts = [(i, prompt, []) for (i, prompt) in current_prompts]
                outputs, all_generated_text = batch_vllm_glyph(glyph_prompts, llm, args, stop_words, args.model_name_or_path, data_name)

        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif ("boxed" not in output and output.endswith("```")):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, args.prompt_type, data_name, use_glyph=args.use_glyph) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=False, use_glyph=args.use_glyph)

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    
    result_json['time_use_in_second'] = time_use
    result_json['time_use_avg_in_second'] = time_use / (len(examples) * args.n_sampling)
    result_json['time_use_in_minite'] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)

    if args.use_glyph:
        tokenizer = AutoTokenizer.from_pretrained(
                        args.model_name_or_path,
                        use_fast=True,
                        trust_remote_code=True
                    )
        def count_tokens(list_str, tokenizer, sep="\n"):
            text = sep.join(list_str)
            return len(tokenizer.encode(text, add_special_tokens=False))
        token_lens = [count_tokens(generated_text, tokenizer) for generated_text in all_generated_text]
        # result_json['token_lens'] = token_lens
        result_json['avg_token_len'] = sum(token_lens) / len(token_lens)
        with open(out_file.replace(".jsonl", f"_{args.prompt_type}_all_generated_text.jsonl"), "w") as f:
            for i, generated_text in enumerate(all_generated_text):
                f.write(json.dumps({"idx": i, "generated_text": generated_text, "token_len": token_lens[i]}, ensure_ascii=False) + "\n")
    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
