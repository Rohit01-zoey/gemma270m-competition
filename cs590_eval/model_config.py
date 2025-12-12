"""Model configuration for router-based pipeline."""

# Configure which model to use for each task type
# Can be HuggingFace model names or local checkpoint paths
# MODEL_MAP = {
#     "reasoning": "google/gemma-3-270m-it",
#     "factual_qa": "google/gemma-3-270m",
#     "instruction_following": "google/gemma-3-270m-it",
# }

# Final use
# MODEL_MAP = {
#     "reasoning": "/usr/xtmp/fj52/590_LLM_project/gemma270m-competition/src/fft/sft_output/checkpoint-85000",
#     "factual_qa": "google/gemma-3-270m",
#     "instruction_following": "/usr/xtmp/fj52/590_LLM_project/gemma270m-competition/src/fft/sft_output/checkpoint-85000",
# }


# Jay checkpoint:
MODEL_MAP = {
    "reasoning": "/usr/xtmp/fj52/590_LLM_project/gemma270m-competition/jay_checkpoints",
    "factual_qa": "google/gemma-3-270m",
    "instruction_following": "/usr/xtmp/fj52/590_LLM_project/gemma270m-competition/src/fft/sft_output/checkpoint-85000",
}