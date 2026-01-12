
# Pretraining datasets base path mapping
DATASET_PREFIX = {
    "Nemotron":"/wl_intelligent/shenzhiwei/model_data/nvidia_distributed2",
    "dolma3": "/wl_intelligent/shenzhiwei/model_data/allen-ai_distributed",
}

# Pretraining datasets mapping， each key corresponds to a group of datasets paths in the value list
PRETRAINING_DATASETS = {
  "crawl-high": [
    "Nemotron-CC-v2.1/High-Quality",
    "Nemotron-CC-v2.1/High-Quality-DQA",
    "Nemotron-CC-v2.1/High-Quality-Translated-To-English",
    "Nemotron-CC-v2/High-Quality"
  ],
  "crawl-medium-high": [
    "Nemotron-CC-v2.1/Medium-High-Quality",
    "Nemotron-CC-v2.1/Medium-High-Quality-Translated-To-English",
    "Nemotron-CC-v2/Medium-High-Quality"
  ],
  "crawl-medium": [
    "Nemotron-CC-v2.1/Medium-Quality",
    "Nemotron-CC-v2/Medium-Quality"
  ],
  "syn-crawl-high": [
    "Nemotron-CC-v2.1/High-Quality-Synthetic",
    "Nemotron-CC-v2.1/High-Quality-Translated-To-English-Synthetic",
    "Nemotron-CC-v2/High-Quality-Synthetic"
  ],
  "syn-crawl-medium-high": [
    "Nemotron-CC-v2.1/Medium-High-Quality-Synthetic"
  ],
  "cc-code": [
    "Nemotron-CC-Code-v1"
  ],
  "code-sft": [
    "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding",
    "Nemotron-Pretraining-SFT-v1/Nemotron-SFT-Code"
  ],
  "stem-sft": [
    "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-RQA",
    "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-InfiniByte-Reasoning",
    "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks",
    "Nemotron-Pretraining-SFT-v1/Nemotron-SFT-MATH"
  ],
  "general-sft": [
    "Nemotron-Pretraining-SFT-v1/Nemotron-SFT-General"
  ],
  "multilingual": [
    "Nemotron-CC-v2/Translated-Diverse-QA"
  ],
#   "crawl++": [
#     "consists of web-crawl derivatives like OpenWebText, BigScience and Reddit"
#   ],
  "academic": [
    # "Nemotron-Pretraining-SFT-v1",
    "Nemotron-CC-v2/Diverse-QA",
    # "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks",
    # "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-RQA",
    # "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding"
  ],
  "code": [
    "Nemotron-Pretraining-Code-v2",
    "Nemotron-Pretraining-Code-v1",
    "dolma3_mix-6T-1025-7B"
  ],
  "math": [
    "Nemotron-CC-Math-v1"
  ],
  "wiki": [
    "Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Wiki-Rewrite"
  ]
}

# DATA_RATIO = {
#     "syn-crawl-high": 0.204,
#     "code": 0.140,
#     "syn-crawl-medium-high": 0.117,
#     "stem-sft": 0.111,
#     "crawl-medium": 0.068,
#     "crawl-high": 0.065,
#     "math": 0.064,
#     "crawl-medium-high": 0.057,
#     "multilingual": 0.050,
#     "academic": 0.041,
#     "code-sft": 0.033,
#     "crawl++": 0.029,
#     "nemotron-cc-code": 0.013,
#     "wiki": 0.006,
#     "general-sft": 0.002,
# }


total_size_in_TB = 0.5
PRETRAINING_DATASET_RATIOS = {
    "crawl-high": 0.065,
    "crawl-medium-high": 0.057,
    "crawl-medium": 0.068,
    "syn-crawl-high": 0.204,
    "syn-crawl-medium-high": 0.117,
    "cc-code": 0.013,          # ⚠️ 图中是 nemotron-cc-code (1.3%)，名字不一致
    "code-sft": 0.033,
    "stem-sft": 0.111,
    "general-sft": 0.002,
    "multilingual": 0.050,
    "crawl++": 0.029,       # ⚠️ 你这里注释掉了，图中存在
    "academic": 0.041,
    "code": 0.140,
    "math": 0.064,
    "wiki": 0.006,
}

