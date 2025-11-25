"""Model configuration for router-based pipeline."""

# Configure which model to use for each task type
# Can be HuggingFace model names or local checkpoint paths
MODEL_MAP = {
    "reasoning": "google/gemma-3-270m-it",
    "factual_qa": "google/gemma-3-270m",
    "instruction_following": "google/gemma-3-270m-it",
}

# Example: Use different models per task
# MODEL_MAP = {
#     "reasoning": "checkpoints/arc_model/final",
#     "factual_qa": "checkpoints/trivia_model/final",
#     "instruction_following": "checkpoints/ifeval_model/final",
# }
