# Model list update

AVAILABLE_MODELS = [
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",  # New model added
    ...
]

model_quota_groups = {
    "claude-sonnet-4.5": {
        "thinking": 10,
    },
    "claude-sonnet-4.6": {  # New model's quota group
        "thinking": 5,
    },
    ...
}