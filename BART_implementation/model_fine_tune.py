import torch
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

def fine_tune_model(article_text, summary_text, epochs=10, learning_rate=1e-5, gradient_clip_val=1.0):
    """
    Fine-tunes the BART model on the provided article and summary.

    Args:
        article_text (str): The article text to fine-tune on.
        summary_text (str): The human-generated summary text to fine-tune on.
        epochs (int): Number of epochs for fine-tuning.
        learning_rate (float): Learning rate for the optimizer.
        gradient_clip_val (float): Maximum value for gradient clipping to prevent gradient explosion.

    Returns:
        model (BartForConditionalGeneration): The fine-tuned BART model.
        tokenizer (BartTokenizer): The tokenizer associated with the BART model.
    """
    # Load the pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the input text (article) and the labels (summary)
    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)
    labels = tokenizer(summary_text, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)

    # Move model and input data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    labels['input_ids'] = labels['input_ids'].to(device)

    # Define the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Set up learning rate schedule with warm-up
    total_steps = epochs * len(inputs['input_ids'])  # Total training steps (epochs * number of batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Initialize variables for tracking best loss and early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 3  # Early stopping after 3 epochs of no improvement

    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass to compute loss
        outputs = model(**inputs, labels=labels['input_ids'])
        loss = outputs.loss
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Backward pass and gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        # Optimizer and learning rate scheduler step
        optimizer.step()
        scheduler.step()

        # Early stopping condition: If loss is not improving, stop training
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Fine-tuning complete.")

    return model, tokenizer
