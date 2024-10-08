import logging
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import re
import evaluate  # Importing the new evaluate module for ROUGE, BLEU, and SacreBLEU
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud  # Importing WordCloud for word clouds
import seaborn as sns  # For heatmap visualization
import numpy as np
import nltk

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure necessary NLTK data is available
nltk.download('punkt')

class TextSummarizer:
    def __init__(self, model_name: str = "google/pegasus-xsum"):
        """
        Initialize the text summarizer with a specific model.
        :param model_name: Name of the model to use for summarization.
        """
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.model.to(device)  # Move model to GPU if available
        self.stop_words = set(stopwords.words('english'))  # Stopwords for cleaning
        self.rouge = evaluate.load("rouge")  # Initialize ROUGE for evaluation
        self.bleu = evaluate.load("bleu")  # Initialize BLEU for evaluation
        self.sacrebleu = evaluate.load("sacrebleu")  # Initialize SacreBLEU for evaluation

    def load_model_and_tokenizer(self):
        """
        Load the pre-trained Pegasus model and tokenizer.
        :return: Model and Tokenizer objects.
        """
        try:
            logging.info(f"Loading model {self.model_name}...")
            tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
            logging.info(f"Model {self.model_name} loaded successfully.")
            return model, tokenizer
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise RuntimeError("Error loading model or tokenizer.") from e

    def clean_summary(self, summary: str) -> str:
        """
        Clean the summary by removing filler words and unnecessary phrases.
        :param summary: Generated summary text.
        :return: Cleaned summary.
        """
        logging.info("Cleaning summary to remove filler words...")
        filler_phrases = [
            r'\b(for more than a decade|as a consequence|writes [\w\s]+,)\b',
            r'\b(according to [\w\s]+|in the opinion of)\b'
        ]
        for phrase in filler_phrases:
            summary = re.sub(phrase, '', summary)

        # Remove excessive whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary

    def summarize_text(self, text: str, prompt: str, min_length: int = 100, max_length: int = 150) -> str:
        """
        Generate a fine-tuned summary for the provided text.
        :param text: The input text to summarize.
        :param prompt: The prompt to guide the summary.
        :param min_length: Minimum length of the summary.
        :param max_length: Maximum length of the summary.
        :return: Cleaned, summarized text.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            logging.error("Invalid input: Input text must be a non-empty string.")
            raise ValueError("Input text must be a non-empty string.")

        # Incorporate the thematic prompt into the input
        text_with_prompt = f"{prompt}: {text}"

        # Tokenize and summarize the text
        logging.info("Tokenizing input text for summarization.")
        inputs = self.tokenizer(text_with_prompt, truncation=True, padding="longest", return_tensors="pt").to(device)

        logging.info("Generating fine-tuned summary...")
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=5,
            no_repeat_ngram_size=3,
            temperature=0.5,
            do_sample=True,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.info("Summary generated successfully. Now cleaning up the text.")
        clean_summary = self.clean_summary(summary)
        return clean_summary

    def evaluate_summary(self, generated_summary: str, reference_summary: str) -> dict:
        """
        Evaluate the generated summary using ROUGE, BLEU, and SacreBLEU metrics.
        :param generated_summary: The summary generated by the model.
        :param reference_summary: The reference (ground-truth) summary.
        :return: Dictionary of evaluation scores (ROUGE, BLEU, SacreBLEU).
        """
        # Clean the summaries before evaluation
        generated_summary = self.clean_summary(generated_summary)
        reference_summary = self.clean_summary(reference_summary)

        # Compute ROUGE scores
        rouge_scores = self.rouge.compute(predictions=[generated_summary], references=[reference_summary])

        # Compute BLEU score
        bleu_scores = self.bleu.compute(predictions=[generated_summary], references=[[reference_summary]])

        # Compute SacreBLEU score
        sacrebleu_scores = self.sacrebleu.compute(predictions=[generated_summary], references=[[reference_summary]])

        return {
            "ROUGE-1 (Precision, Recall, F1)": rouge_scores["rouge1"],
            "ROUGE-2 (Precision, Recall, F1)": rouge_scores["rouge2"],
            "ROUGE-L (Precision, Recall, F1)": rouge_scores["rougeL"],  # Corrected to "rougeL"
            "BLEU": bleu_scores["bleu"],
            "SacreBLEU": sacrebleu_scores["score"]
        }

    def plot_token_level_overlap_heatmap(self, generated_summary: str, reference_summary: str):
        """
        Plot a heatmap showing token-level overlap between the generated summary and reference summary.
        :param generated_summary: The summary generated by the model.
        :param reference_summary: The reference (ground-truth) summary.
        """
        # Tokenize both summaries
        gen_tokens = nltk.word_tokenize(generated_summary)
        ref_tokens = nltk.word_tokenize(reference_summary)

        # Initialize an empty matrix
        overlap_matrix = np.zeros((len(gen_tokens), len(ref_tokens)))

        # Fill the matrix with 1s where tokens match, 0 otherwise
        for i, gen_token in enumerate(gen_tokens):
            for j, ref_token in enumerate(ref_tokens):
                if gen_token.lower() == ref_token.lower():
                    overlap_matrix[i, j] = 1

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, cmap='Blues', xticklabels=ref_tokens, yticklabels=gen_tokens, cbar=False)
        plt.title('Token-Level Overlap Heatmap')
        plt.xlabel('Reference Summary Tokens')
        plt.ylabel('Generated Summary Tokens')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()

    def plot_word_overlap(self, generated_summary: str, reference_summary: str):
        """
        Plot word overlap between the generated summary and reference summary using a Venn diagram.
        :param generated_summary: The summary generated by the model.
        :param reference_summary: The reference (ground-truth) summary.
        """
        # Tokenize summaries
        gen_words = set(nltk.word_tokenize(generated_summary.lower()))
        ref_words = set(nltk.word_tokenize(reference_summary.lower()))

        # Create Venn diagram
        plt.figure(figsize=(6, 6))
        venn2([gen_words, ref_words], set_labels=("Generated Summary", "Reference Summary"))
        plt.title("Word Overlap between Generated and Reference Summary")
        plt.show()

    def plot_ngram_overlap(self, generated_summary: str, reference_summary: str, n: int = 2):
        """
        Plot n-gram overlap between generated and reference summary using a bar plot.
        :param generated_summary: The summary generated by the model.
        :param reference_summary: The reference (ground-truth) summary.
        :param n: n-gram size to analyze (default is 2 for bigrams).
        """
        def get_ngrams(text, n):
            vectorizer = CountVectorizer(ngram_range=(n, n)).fit([text])
            ngrams = vectorizer.get_feature_names_out()
            return Counter(ngrams)

        gen_ngrams = get_ngrams(generated_summary, n)
        ref_ngrams = get_ngrams(reference_summary, n)

        # Find common n-grams
        all_ngrams = set(gen_ngrams.keys()).union(set(ref_ngrams.keys()))

        gen_counts = [gen_ngrams[ng] for ng in all_ngrams]
        ref_counts = [ref_ngrams[ng] for ng in all_ngrams]

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        index = range(len(all_ngrams))
        plt.bar(index, gen_counts, width=0.4, label="Generated", align='center')
        plt.bar(index, ref_counts, width=0.4, label="Reference", align='edge')
        plt.xticks(index, all_ngrams, rotation=90)
        plt.title(f"{n}-Gram Overlap between Generated and Reference Summary")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def plot_word_clouds(self, generated_summary: str, reference_summary: str):
        """
        Plot word clouds for the generated and reference summaries.
        :param generated_summary: The summary generated by the model.
        :param reference_summary: The reference (ground-truth) summary.
        """
        # Generate word clouds
        generated_wordcloud = WordCloud(background_color='white', colormap='Blues').generate(generated_summary)
        reference_wordcloud = WordCloud(background_color='white', colormap='Greens').generate(reference_summary)

        # Plot the word clouds
        plt.figure(figsize=(14, 6))

        # Generated summary word cloud
        plt.subplot(1, 2, 1)
        plt.imshow(generated_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Generated Summary Word Cloud")

        # Reference summary word cloud
        plt.subplot(1, 2, 2)
        plt.imshow(reference_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Reference Summary Word Cloud")

        plt.show()

def setup_logging():
    """
    Setup logging configuration to track application behavior.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("summarizer.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main function to execute the summarizer.
    """
    setup_logging()

    # Define the prompt
    prompt = "Summarize the article below"

    # Example usage: single article summarization
    article_text = """
China’s economy picked up pace in the first quarter as Beijing’s plan to boost growth by pouring money into factories began to show results.
But that approach is leading to a lopsided recovery and stoking trade tensions overseas, with Western governments and some big emerging economies crying foul over a growing wave of cheap Chinese imports they say threatens domestic jobs and industries.
With familiar signs of weakness in consumption and real ­estate in the first three months of the year, many economists say Beijing still isn’t doing enough to support households and nurture a more balanced recovery.
And the loss of some momentum in March compared with the preceding two months reinforced expectations that further stimulus will be needed to ensure that the government meets its growth target of 5 per cent for the year.
China said its economy grew 5.3 per cent in the first quarter compared with the same three months a year earlier, a faster pace than the 5.2 per cent year-over-year growth rate that the country notched in the final quarter of 2023, China’s National Bureau of Statistics said on Tuesday.
The pick-up was propelled by a rise in industrial production and swelling investment in factories. After a challenging few years, Chinese officials are steering activity and investment towards manufacturing and exports to compensate for domestic consumers’ reluctance to spend and a continuing crunch in the property market.
Beijing is also seeking to stake out a commanding lead in newer hi-tech industries such as electric vehicles and renewable energy equipment – sectors it counts among the “new productive ­forces” it wants to harness to fuel the next stage of China’s economic ascendancy.
But Beijing’s strategy is raising hackles around the world as governments baulk at the risk to jobs and industries from a potential rerun of the “China shock” of the early 2000s, when a torrent of Chinese imports hit low-tech manufacturing in the US, costing the country an estimated two million jobs.
The US and Europe are pushing back against Chinese EVs, solar panels and wind turbines, new industries that they are also seeking to dominate. Emerging economies are feeling the heat from China’s manufacturing glut too, with Brazil, India and Mexico among those investigating whether Chinese products such as steel and ceramics are being dumped on their markets at unfairly low prices.
China says its companies are competing fairly and has criticised such moves as protectionism. The International Monetary Fund and others warn that these mounting tensions over trade could lead to the global economy fracturing, with blocs of countries allied around the US and China, respectively, and broader trade impeded.
Tuesday’s data laid out in detail the fruits of Beijing’s strategy, with industrial production rising 6.1 per cent from a year earlier in the first quarter, propelling overall growth. Investment in manufacturing rose 9.9 per cent.
But there were also signs of the strategy’s limits. There was a growing mismatch between ballooning supply and lacklustre demand, with China’s factories reporting a fall in the amount of available production capacity they are using. Overall capacity utilisation fell 0.7 percentage points in the first quarter to 73.6 per cent, with steeper drops in industries including cars and electrical machinery. In February, inventories of finished products were 2.4 per cent larger than a year earlier. “It is a positive omen for the world economy that China seems to be getting past a rough patch. However, these data will not ­assuage concerns that a production-led recovery and weak consumption demand could lead China to aggressively push exports to keep its recovery going,” said Eswar Prasad, professor of trade policy and economics at Cornell University and a former head of the IMF’s China division.
"""

    ground_truth_summary = """
    After a challenging few years, Chinese officials are steering activity and investment towards manufacturing and exports to compensate for domestic consumers’ reluctance to spend and a continuing crunch in the property market. As a result, China’s economy picked up pace in the first quarter, propelled by a rise in industrial production and swelling investment in factories. However this approach is leading to a lopsided recovery. Domestically, with familiar signs of weakness in consumption and real ­estate in the first three months of the year, many economists say Beijing still isn’t doing enough to support households and nurture a more balanced recovery. Outside China, Western governments and some big emerging economies are crying foul over a growing wave of cheap Chinese imports they say threaten domestic jobs and industries, while the International Monetary Fund and others warn that these mounting tensions over trade could lead to the global economy fracturing, with blocs of countries allied around the US and China, respectively.
    """

    try:
        summarizer = TextSummarizer(model_name="google/pegasus-xsum")
        summary = summarizer.summarize_text(article_text, prompt)
        print("Prompt: ", prompt)
        print("Generated Cleaned Summary:")
        print(summary)

        # Evaluate the generated summary
        scores = summarizer.evaluate_summary(summary, ground_truth_summary)
        print("\nEvaluation Scores:")
        for metric, score in scores.items():
            print(f"{metric}: {score}")

        # Visualizations
        summarizer.plot_word_overlap(summary, ground_truth_summary)
        summarizer.plot_ngram_overlap(summary, ground_truth_summary, n=2)
        summarizer.plot_word_clouds(summary, ground_truth_summary)
        summarizer.plot_token_level_overlap_heatmap(summary, ground_truth_summary)  # Added Token-Level Heatmap

    except Exception as e:
        logging.error(f"Error during summarization: {e}")

if __name__ == "__main__":
    main()
