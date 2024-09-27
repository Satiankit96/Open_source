from data_loader import load_data, preprocess_text
from model_fine_tune import fine_tune_model
from diversity_control import apply_top_k_sampling, apply_nucleus_sampling
from evaluation import calculate_rouge, calculate_bert_score, calculate_bleu
import pandas as pd
import config

def main():
    # Step 1: Load and preprocess the data
    article_text, summary_text = load_data()
    article_text = preprocess_text(article_text)
    summary_text = preprocess_text(summary_text)

    # Step 2: Fine-tune the model on the provided data
    print("Starting model fine-tuning...")
    model, tokenizer = fine_tune_model(article_text, summary_text, epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE)
    print("Fine-tuning complete.")

    # Step 3: Finalize summaries with Top-k and Nucleus Sampling techniques
    print("\nGenerating final summaries using refined Top-k and Nucleus Sampling...")

    # Final refined Top-k Sampling
    final_top_k_summary = apply_top_k_sampling(model, tokenizer, article_text, top_k=30)
    print(f"Final Top-k Sampling Summary:\n{final_top_k_summary}")

    # Final refined Nucleus Sampling
    final_nucleus_summary = apply_nucleus_sampling(model, tokenizer, article_text, top_p=0.85)
    print(f"Final Nucleus Sampling Summary:\n{final_nucleus_summary}")

    # Step 4: Final Evaluation of the best summaries
    print("\nFinal Evaluation of Refined Summaries...")

    # Evaluate final top-k summary
    final_rouge_top_k = calculate_rouge(final_top_k_summary, summary_text)
    final_bert_score_top_k = calculate_bert_score(final_top_k_summary, summary_text)
    final_bleu_top_k = calculate_bleu(final_top_k_summary, summary_text)
    print(f"Final Top-k Sampling Evaluation - ROUGE: {final_rouge_top_k}, BERTScore: {final_bert_score_top_k}, BLEU: {final_bleu_top_k}")

    # Evaluate final nucleus summary
    final_rouge_nucleus = calculate_rouge(final_nucleus_summary, summary_text)
    final_bert_score_nucleus = calculate_bert_score(final_nucleus_summary, summary_text)
    final_bleu_nucleus = calculate_bleu(final_nucleus_summary, summary_text)
    print(f"Final Nucleus Sampling Evaluation - ROUGE: {final_rouge_nucleus}, BERTScore: {final_bert_score_nucleus}, BLEU: {final_bleu_nucleus}")

    # Step 5: Save the final scores
    results = {
        "Sampling Method": ["Top-k", "Nucleus"],
        "ROUGE": [final_rouge_top_k, final_rouge_nucleus],
        "BERTScore": [final_bert_score_top_k, final_bert_score_nucleus],
        "BLEU": [final_bleu_top_k, final_bleu_nucleus]
    }
    df_results = pd.DataFrame(results)
    df_results.to_excel('final_summary_evaluation.xlsx', index=False)
    print("\nFinal evaluation scores saved to 'final_summary_evaluation.xlsx'.")

if __name__ == "__main__":
    main()
