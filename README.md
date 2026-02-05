<div align="center">

<h1> VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.


<h5 align="center"> 

<a href='https://arxiv.org/abs/2601.22069'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/yiboowang/VTC-R1-Glyph'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href=''><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>


[Yibo Wang]()<sup>1</sup>,
[Yongcheng Jing]()<sup>1</sup>,
[Shunyu Liu]()<sup>1</sup>,
[Hao Guan]()<sup>1</sup>,
[Rong-Cheng Tu]()<sup>1</sup>,

[Chengyu Wang]()<sup>2</sup>,
[Jun Huang]()<sup>2</sup>,
[Dacheng Tao]()<sup>1</sup>



<sup>1</sup>[Nanyang Technological University](https://www.ntu.edu.sg/), <sup>2</sup>[Alibaba Cloud Computing]()



</h5>
</div>





## News

- [x] **`Jan 30, 2026.`** We release our paper in [arxiv](https://arxiv.org/abs/2601.22069) and our model in [huggingface](https://huggingface.co/yiboowang/VTC-R1-Glyph).

## Case Inference


### Setup
First, please install the required dependencies using the following command:
```bash
apt-get install poppler-utils # or conda install -c conda-forge poppler
pip install torch==2.6.0
pip install transformers==4.57.1 
pip install reportlab
pip install pdf2image 
```
We provide the inference code for running VTC-R1 models, which can output VTC-R1 style reasoning.

```bash
python inference.py # replace your model path
```


## [vLLM](https://github.com/vllm-project/vllm) for Batch Inference (Evaluation)

### Setup
We recommend creating a new environment, such as conda, to avoid dependency conflicts.
```bash
apt-get install poppler-utils # or conda install -c conda-forge poppler
pip install torch==2.8.0
pip install transformers==4.57.1 
pip install vllm==0.10.2
cd evaluation
pip install -r requirements.txt
```
Then, run the code to evaluation:

```bash
bash scripts/run_eval.sh direct /YOUR/PATH/TO/MODEL # gsm8k, math500, gpqa_d

bash scripts/run_eval_16.sh direct /YOUR/PATH/TO/MODEL # aime25, amc23
```

## Training
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune the VTC-R1 models. We provide the training instructions and configs here.

First, install LLaMA-Factory according to the [official_instruction](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation).

Then, refer [here](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) and update the following customized dataset into `dataset_info.json` in LLaMA-Factory.
```bash
"vtc-r1": {
    "file_name": "./openr1_vtc_r1",
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images": "images"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
    }
  },
```

Finally, you can use the following command to train the models.
```bash
llamafactory-cli train examples/vtcr1_glyph_full_sft.yaml
```





## Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@article{wang2026vtc,
  title={VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning},
  author={Wang, Yibo and Jing, Yongcheng and Liu, Shunyu and Guan, Hao and Tu, Rong-cheng and Wang, Chengyu and Huang, Jun and Tao, Dacheng},
  journal={arXiv preprint arXiv:2601.22069},
  year={2026}
}
```


## Acknowledgment
Our work is primarily based on the following codebases. We are sincerely grateful for their work.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use llama-factory to fine-tune Models.
- [Glyph](https://github.com/thu-coai/Glyph): We use Glyph as the intial ckpt and rendering.
- [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness) We use it for evaluation.
