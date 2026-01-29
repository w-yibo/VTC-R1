import pandas as pd

# 路径
parquet_path = "/home/wyb/project/math-evaluation-harness/data/gpqa_d/gpqa_diamond.parquet"
jsonl_path = "/home/wyb/project/math-evaluation-harness/data/gpqa_d/test.jsonl"

# 读取 parquet
df = pd.read_parquet(parquet_path)

# 写成 jsonl（一行一个 JSON）
df.to_json(
    jsonl_path,
    orient="records",
    lines=True,
    force_ascii=False  # 保留中文
)

print(f"Converted {len(df)} rows to {jsonl_path}")
