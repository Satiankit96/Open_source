import pandas as pd
from tqdm import tqdm
from evaluation import calculate_rouge, calculate_bert_score, calculate_bleu

def generate_summary(model, tokenizer, article_text, num_beams=4, length_penalty=1.0, repetition_penalty=1.0):
    """
    Generates a summary using the provided model and parameters.

    Args:
        model (BartForConditionalGeneration): The fine-tuned BART model.
        tokenizer (BartTokenizer): The tokenizer associated with the BART model.
        article_text (str): The article text to summarize.
        num_beams (int): Beam search width for diversity (default: 4).
        length_penalty (float): Penalty for summary length (default: 1.0).
        repetition_penalty (float): Penalty to reduce repetition in the summary (default: 1.0).

    Returns:
        str: The generated summary.
    """
    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')

    # Generate summary using beam search and other parameters
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        max_length=150,  # Adjust these based on your requirements
        min_length=50,
        no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def tune_inference_parameters(model, tokenizer, article_text, ground_truth_summary):
    """
    Tunes different inference parameters (num_beams, length_penalty, repetition_penalty) and evaluates
    performance using ROUGE, BERTScore, and BLEU.

    Args:
        model (BartForConditionalGeneration): The fine-tuned BART model.
        tokenizer (BartTokenizer): The tokenizer associated with the BART model.
        article_text (str): The article text to summarize.
        ground_truth_summary (str): The human-generated summary to compare against.

    Returns:
        tuple: The best parameter combination and the best ROUGE score.
    """
    # Define the range of parameters to tune
    num_beams_range = [4, 6, 8, 10]
    length_penalty_range = [0.8, 1.0, 1.2, 1.5]
    repetition_penalty_range = [1.0, 1.5, 2.0]

    results = []
    best_rouge = 0
    best_params = None

    total_iterations = len(num_beams_range) * len(length_penalty_range) * len(repetition_penalty_range)

    # Use tqdm to track progress
    with tqdm(total=total_iterations, desc="Tuning Inference Parameters") as pbar:
        for num_beams in num_beams_range:
            for length_penalty in length_penalty_range:
                for repetition_penalty in repetition_penalty_range:
                    # Generate summary using current parameter combination
                    summary = generate_summary(model, tokenizer, article_text, num_beams, length_penalty, repetition_penalty)

                    # Calculate evaluation metrics
                    rouge_score = calculate_rouge(summary, ground_truth_summary)
                    bert_f1 = calculate_bert_score(summary, ground_truth_summary)
                    bleu_score = calculate_bleu(summary, ground_truth_summary)

                    # Log progress
                    results.append({
                        'num_beams': num_beams,
                        'length_penalty': length_penalty,
                        'repetition_penalty': repetition_penalty,
                        'rouge_score': rouge_score,
                        'bert_f1': bert_f1,
                        'bleu_score': bleu_score
                    })

                    # Track the best parameter configuration based on ROUGE score
                    if rouge_score > best_rouge:
                        best_rouge = rouge_score
                        best_params = (num_beams, length_penalty, repetition_penalty)

                    # Update progress bar
                    pbar.update(1)

    # Save results to an Excel file for analysis
    df = pd.DataFrame(results)
    df.to_excel('inference_results.xlsx', index=False)

    # Print and return the best parameter combination and best ROUGE score
    print(f"Best parameters: num_beams={best_params[0]}, length_penalty={best_params[1]}, repetition_penalty={best_params[2]}")
    print(f"Best ROUGE-1 Score: {best_rouge:.4f}")

    return best_params, best_rouge
