"""
File specifying the top-3 model prompts ({model_prompts}), and cateegory prompts ({category_prompts}), per LLM. 
"""
model_prompts = {
    "Qwen/Qwen2.5-32B-Instruct" : [
        "T:q|R:-|OG:i|PI:s",
        "T:q|R:-|OG:i|PI:-",
        "T:i|R:-|OG:i|PI:s"
    ],
    "Qwen/Qwen3-8B" : [
        "T:q|R:c|OG:d|PI:d",
        "T:q|R:-|OG:d|PI:s",
        "T:i|R:c|OG:d|PI:d"
    ],
    "Qwen/Qwen3-30B-A3B" : [
        "T:q|R:-|OG:-|PI:d",
        "T:i|R:c|OG:-|PI:s",
        "T:q|R:c|OG:d|PI:s"
    ], 
    "google/gemma-4-31B-it" : [
        "T:i|R:-|OG:d|PI:d",
        "T:i|R:c|OG:d|PI:d",
        "T:i|R:-|OG:d|PI:-"
    ],
    "google/gemma-4-26B-A4B-it" : [
        "T:q|R:-|OG:d|PI:s",
        "T:i|R:-|OG:d|PI:-",
        "T:q|R:-|OG:d|PI:-"
    ]
}

category_prompts = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "0": [
            "T:i|R:-|OG:-|PI:d",
            "T:i|R:-|OG:d|PI:-",
            "T:q|R:-|OG:d|PI:-"
        ],
        "2": [
            "T:i|R:c|OG:i|PI:d",
            "T:i|R:c|OG:d|PI:-",
            "T:i|R:c|OG:d|PI:d"
        ],
        "3": [
            "T:i|R:c|OG:i|PI:s",
            "T:i|R:c|OG:i|PI:d",
            "T:i|R:-|OG:i|PI:d"
        ],
        "4": [
            "T:q|R:-|OG:i|PI:-",
            "T:q|R:c|OG:i|PI:d",
            "T:q|R:c|OG:i|PI:-"
        ],
        "7": [
            "T:q|R:-|OG:d|PI:s",
            "T:q|R:c|OG:d|PI:s",
            "T:q|R:c|OG:i|PI:s"
        ],
        "8": [
            "T:q|R:-|OG:i|PI:-",
            "T:q|R:-|OG:i|PI:d",
            "T:i|R:-|OG:d|PI:-"
        ]
    },
    "Qwen/Qwen3-8B": {
        "0": [
            "T:i|R:-|OG:d|PI:d",
            "T:q|R:c|OG:d|PI:-",
            "T:q|R:c|OG:d|PI:s"
        ],
        "2": [
            "T:q|R:c|OG:d|PI:d",
            "T:q|R:-|OG:d|PI:d",
            "T:q|R:-|OG:d|PI:s"
        ],
        "3": [
            "T:q|R:c|OG:d|PI:s",
            "T:i|R:c|OG:d|PI:-",
            "T:q|R:c|OG:-|PI:d"
        ],
        "4": [
            "T:i|R:c|OG:d|PI:-",
            "T:q|R:c|OG:d|PI:d",
            "T:i|R:c|OG:i|PI:d"
        ],
        "7": [
            "T:i|R:-|OG:-|PI:d",
            "T:i|R:c|OG:d|PI:d",
            "T:i|R:c|OG:-|PI:d"
        ],
        "8": [
            "T:q|R:-|OG:d|PI:s",
            "T:q|R:-|OG:d|PI:-",
            "T:i|R:-|OG:d|PI:s"
        ]
    },
    "Qwen/Qwen3-30B-A3B": {
        "0": [
            "T:i|R:c|OG:-|PI:d",
            "T:q|R:c|OG:d|PI:s",
            "T:i|R:-|OG:d|PI:-"
        ],
        "2": [
            "T:q|R:-|OG:-|PI:d",
            "T:q|R:c|OG:d|PI:-",
            "T:q|R:c|OG:-|PI:d"
        ],
        "3": [
            "T:i|R:-|OG:-|PI:-",
            "T:i|R:c|OG:d|PI:s",
            "T:q|R:c|OG:d|PI:s"
        ],
        "4": [
            "T:i|R:c|OG:i|PI:-",
            "T:i|R:c|OG:-|PI:s",
            "T:q|R:c|OG:-|PI:s"
        ],
        "7": [
            "T:q|R:-|OG:-|PI:d",
            "T:q|R:c|OG:-|PI:s",
            "T:q|R:c|OG:-|PI:d"
        ],
        "8": [
            "T:q|R:c|OG:d|PI:d",
            "T:i|R:c|OG:-|PI:s",
            "T:i|R:-|OG:-|PI:-"
        ]
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "0": [
            "T:i|R:c|OG:d|PI:-",
            "T:i|R:-|OG:d|PI:-",
            "T:i|R:c|OG:d|PI:d"
        ],
        "2": [
            "T:q|R:-|OG:i|PI:s",
            "T:q|R:-|OG:d|PI:s",
            "T:i|R:-|OG:-|PI:d"
        ],
        "3": [
            "T:q|R:-|OG:i|PI:s",
            "T:q|R:c|OG:i|PI:s",
            "T:i|R:-|OG:i|PI:d"
        ],
        "4": [
            "T:q|R:c|OG:d|PI:d",
            "T:q|R:-|OG:d|PI:s",
            "T:q|R:c|OG:i|PI:d"
        ],
        "7": [
            "T:q|R:-|OG:i|PI:s",
            "T:i|R:-|OG:i|PI:s",
            "T:q|R:c|OG:i|PI:s"
        ],
        "8": [
            "T:q|R:-|OG:d|PI:d",
            "T:i|R:c|OG:d|PI:s",
            "T:q|R:c|OG:i|PI:s"
        ]
    },
    "Qwen/Qwen3-30B-A3B-Instruct": {
        "0": [
            "T:i|R:c|OG:i|PI:-",
            "T:i|R:-|OG:i|PI:-",
            "T:i|R:-|OG:i|PI:d"
        ],
        "2": [
            "T:i|R:c|OG:i|PI:d",
            "T:i|R:c|OG:i|PI:s",
            "T:i|R:c|OG:i|PI:-"
        ],
        "3": [
            "T:i|R:-|OG:i|PI:d",
            "T:i|R:-|OG:i|PI:-",
            "T:i|R:c|OG:i|PI:d"
        ],
        "4": [
            "T:q|R:-|OG:d|PI:s",
            "T:q|R:-|OG:d|PI:d",
            "T:i|R:-|OG:i|PI:d"
        ],
        "7": [
            "T:i|R:-|OG:i|PI:s",
            "T:i|R:-|OG:i|PI:d",
            "T:q|R:-|OG:i|PI:s"
        ],
        "8": [
            "T:i|R:-|OG:i|PI:s",
            "T:i|R:-|OG:i|PI:d",
            "T:q|R:-|OG:i|PI:s"
        ]
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "0": [
            "T:i|R:c|OG:d|PI:d",
            "T:i|R:c|OG:d|PI:s",
            "T:i|R:-|OG:i|PI:-"
        ],
        "2": [
            "T:i|R:-|OG:i|PI:d",
            "T:i|R:-|OG:i|PI:s",
            "T:q|R:-|OG:i|PI:s"
        ],
        "3": [
            "T:q|R:-|OG:i|PI:-",
            "T:i|R:-|OG:d|PI:s",
            "T:q|R:-|OG:d|PI:-"
        ],
        "4": [
            "T:q|R:c|OG:i|PI:-",
            "T:q|R:c|OG:i|PI:d",
            "T:q|R:c|OG:d|PI:-"
        ],
        "7": [
            "T:i|R:c|OG:i|PI:s",
            "T:i|R:-|OG:i|PI:s",
            "T:q|R:-|OG:i|PI:s"
        ],
        "8": [
            "T:q|R:-|OG:i|PI:-",
            "T:q|R:-|OG:i|PI:s",
            "T:q|R:c|OG:i|PI:s"
        ]
    },
    "google/gemma-4-31B-it": {
        "0": [
            "T:i|R:-|OG:d|PI:-",
            "T:q|R:-|OG:i|PI:d",
            "T:q|R:c|OG:i|PI:-"
        ],
        "2": [
            "T:i|R:c|OG:d|PI:-",
            "T:i|R:c|OG:d|PI:d",
            "T:i|R:-|OG:d|PI:-"
        ],
        "3": [
            "T:i|R:-|OG:d|PI:d",
            "T:i|R:-|OG:d|PI:s",
            "T:i|R:c|OG:d|PI:-"
        ],
        "4": [
            "T:i|R:-|OG:i|PI:-",
            "T:i|R:-|OG:d|PI:s",
            "T:i|R:-|OG:d|PI:d"
        ],
        "7": [
            "T:q|R:-|OG:d|PI:s",
            "T:q|R:-|OG:d|PI:d",
            "T:q|R:-|OG:i|PI:d"
        ],
        "8": [
            "T:i|R:-|OG:-|PI:-",
            "T:i|R:c|OG:i|PI:s",
            "T:i|R:-|OG:i|PI:s"
        ]
    },
    "google/gemma-4-26B-A4B-it": {
        "0": [
            "T:q|R:-|OG:d|PI:-",
            "T:i|R:c|OG:d|PI:-",
            "T:i|R:-|OG:d|PI:-"
        ],
        "2": [
            "T:i|R:-|OG:i|PI:s",
            "T:q|R:-|OG:d|PI:s",
            "T:i|R:-|OG:i|PI:-"
        ],
        "3": [
            "T:q|R:-|OG:d|PI:s",
            "T:q|R:-|OG:d|PI:d",
            "T:i|R:c|OG:i|PI:-"
        ],
        "4": [
            "T:q|R:-|OG:d|PI:-",
            "T:q|R:-|OG:i|PI:-",
            "T:i|R:-|OG:-|PI:s"
        ],
        "7": [
            "T:q|R:c|OG:i|PI:s",
            "T:q|R:-|OG:-|PI:s",
            "T:q|R:c|OG:d|PI:s"
        ],
        "8": [
            "T:i|R:-|OG:d|PI:-",
            "T:q|R:-|OG:i|PI:-",
            "T:q|R:c|OG:i|PI:-"
        ]
    } 
}