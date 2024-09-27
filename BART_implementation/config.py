# config.py

# Inference Parameters (used in inference_tuner.py)
NUM_BEAMS = [4, 6, 8, 10, 12]  # Range of num_beams for beam search
LENGTH_PENALTY = [0.8, 1.0, 1.2, 1.5, 2.0]  # Range of length_penalty values
REPETITION_PENALTY = [1.0, 1.5, 2.0]  # Range of repetition_penalty values

# Neural Network Optimizations (used in model_fine_tune.py)
LEARNING_RATE = 1e-5  # Learning rate for fine-tuning the BART model
GRADIENT_CLIP_VALUE = 1.0  # Value for gradient clipping to prevent gradient explosion
EPOCHS = 10  # Number of epochs for fine-tuning

# Sampling Parameters (used in diversity_control.py)
TEMPERATURE = [0.7, 1.0, 1.5]  # Range of temperature values for sampling
TOP_K = [30, 50, 100]  # Top-k values for sampling
TOP_P = [0.85, 0.9, 0.95]  # Top-p (nucleus sampling) values for sampling

# Miscellaneous
MAX_LENGTH = 150  # Maximum length of the generated summary
MIN_LENGTH = 50  # Minimum length of the generated summary
NO_REPEAT_NGRAM_SIZE = 3  # Prevent repetition of 3-grams
EARLY_STOPPING = True  # Enable early stopping during decoding
