def generate_with_temperature(model, tokenizer, article_text, temperature=1.0):
    """
    Generates a summary using a specified temperature to control diversity.

    Args:
        model (BartForConditionalGeneration): The fine-tuned BART model.
        tokenizer (BartTokenizer): The tokenizer associated with the BART model.
        article_text (str): The article text to summarize.
        temperature (float): The temperature value for controlling creativity and diversity (default: 1.0).

    Returns:
        str: The generated summary.
    """
    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')

    # Generate summary using the temperature value for diversity control
    summary_ids = model.generate(
        inputs['input_ids'],
        do_sample=True,  # Enable sampling instead of greedy decoding
        temperature=temperature,  # Controls the randomness of predictions
        max_length=150,
        min_length=50,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def apply_top_k_sampling(model, tokenizer, article_text, top_k=50):
    """
    Generates a summary using top-k sampling to control the number of high-probability tokens to sample from.

    Args:
        model (BartForConditionalGeneration): The fine-tuned BART model.
        tokenizer (BartTokenizer): The tokenizer associated with the BART model.
        article_text (str): The article text to summarize.
        top_k (int): The top-k value for sampling (default: 50).

    Returns:
        str: The generated summary.
    """
    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')

    # Generate summary using top-k sampling for diversity control
    summary_ids = model.generate(
        inputs['input_ids'],
        do_sample=True,  # Enable sampling instead of greedy decoding
        top_k=top_k,  # Restrict sampling to top k most likely next tokens
        max_length=150,
        min_length=50,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def apply_nucleus_sampling(model, tokenizer, article_text, top_p=0.9):
    """
    Generates a summary using nucleus (top-p) sampling, where sampling is restricted to the smallest possible set of tokens
    whose cumulative probability exceeds top_p.

    Args:
        model (BartForConditionalGeneration): The fine-tuned BART model.
        tokenizer (BartTokenizer): The tokenizer associated with the BART model.
        article_text (str): The article text to summarize.
        top_p (float): The top-p value for nucleus sampling (default: 0.9).

    Returns:
        str: The generated summary.
    """
    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')

    # Generate summary using top-p sampling for diversity control
    summary_ids = model.generate(
        inputs['input_ids'],
        do_sample=True,  # Enable sampling instead of greedy decoding
        top_p=top_p,  # Restrict sampling to tokens with cumulative probability > top_p
        max_length=150,
        min_length=50,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
