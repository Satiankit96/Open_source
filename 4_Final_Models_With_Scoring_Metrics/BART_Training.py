import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Generate multiple summaries with different parameters
class BARTSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        tokenizer = BartTokenizer.from_pretrained(self.model_name)
        model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        model.eval()
        return model, tokenizer

    def summarize_text(self, text, num_beams, length_penalty, repetition_penalty, max_length):
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def calculate_metrics(self, generated_summary, ground_truth):
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = rouge.score(ground_truth, generated_summary)['rougeL'].fmeasure
        sacre_bleu = sacrebleu.corpus_bleu([generated_summary], [[ground_truth]]).score
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([ground_truth.split()], generated_summary.split(), smoothing_function=smoothie)
        _, _, bert_f1 = bert_score([generated_summary], [ground_truth], lang="en", model_type="roberta-large")
        return rougeL, sacre_bleu, bleu, bert_f1.mean().item()

def generate_data(article_text, ground_truth, summarizer, n_samples=100):
    data = []
    print("Generating data with randomized parameters and calculating metrics...")

    for _ in tqdm(range(n_samples)):
        # Randomize parameters for each summary generation
        num_beams = np.random.randint(5, 12)
        length_penalty = np.random.uniform(0.5, 2.0)
        repetition_penalty = np.random.uniform(1.0, 2.0)
        max_length = np.random.randint(200, 400)

        # Generate summary
        summary = summarizer.summarize_text(article_text, num_beams, length_penalty, repetition_penalty, max_length)

        # Calculate metrics
        rougeL, sacre_bleu, bleu, bert_f1 = summarizer.calculate_metrics(summary, ground_truth)

        # Store data
        data.append([num_beams, length_penalty, repetition_penalty, max_length, rougeL, sacre_bleu, bleu, bert_f1])

    # Convert to DataFrame
    columns = ['num_beams', 'length_penalty', 'repetition_penalty', 'max_length', 'ROUGE-L', 'SacreBLEU', 'BLEU', 'BERTScore-F1']
    return pd.DataFrame(data, columns=columns)

# Visualize parameter effects on metrics
def visualize_parameters(df):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot num_beams vs metrics
    axs[0, 0].scatter(df['num_beams'], df['ROUGE-L'], color='blue', label='ROUGE-L')
    axs[0, 0].set_title('num_beams vs ROUGE-L')
    axs[0, 0].set_xlabel('num_beams')
    axs[0, 0].set_ylabel('ROUGE-L')

    # Plot length_penalty vs metrics
    axs[0, 1].scatter(df['length_penalty'], df['SacreBLEU'], color='green', label='SacreBLEU')
    axs[0, 1].set_title('length_penalty vs SacreBLEU')
    axs[0, 1].set_xlabel('length_penalty')
    axs[0, 1].set_ylabel('SacreBLEU')

    # Plot repetition_penalty vs metrics
    axs[1, 0].scatter(df['repetition_penalty'], df['BLEU'], color='red', label='BLEU')
    axs[1, 0].set_title('repetition_penalty vs BLEU')
    axs[1, 0].set_xlabel('repetition_penalty')
    axs[1, 0].set_ylabel('BLEU')

    # Plot max_length vs metrics
    axs[1, 1].scatter(df['max_length'], df['BERTScore-F1'], color='purple', label='BERTScore-F1')
    axs[1, 1].set_title('max_length vs BERTScore-F1')
    axs[1, 1].set_xlabel('max_length')
    axs[1, 1].set_ylabel('BERTScore-F1')

    plt.tight_layout()
    plt.show()

# Prepare dataset
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
ground_truth = """
    After a challenging few years, Chinese officials are steering activity and investment towards manufacturing and exports to compensate for domestic consumers’ reluctance to spend and a continuing crunch in the property market. As a result, China’s economy picked up pace in the first quarter, propelled by a rise in industrial production and swelling investment in factories. However this approach is leading to a lopsided recovery. Domestically, with familiar signs of weakness in consumption and real ­estate in the first three months of the year, many economists say Beijing still isn’t doing enough to support households and nurture a more balanced recovery. Outside China, Western governments and some big emerging economies are crying foul over a growing wave of cheap Chinese imports they say threaten domestic jobs and industries, while the International Monetary Fund and others warn that these mounting tensions over trade could lead to the global economy fracturing, with blocs of countries allied around the US and China, respectively.
    """
summarizer = BARTSummarizer()

# Generating data with progress bar
df = generate_data(article_text, ground_truth, summarizer, n_samples=100)

# Display a portion of the generated data
print("Generated data:")
print(df.head())

# Visualize parameter effects on metrics
print("\nVisualizing the effect of parameters on summarization metrics...")
visualize_parameters(df)

# Split the data into train/test sets
X = df[['num_beams', 'length_penalty', 'repetition_penalty', 'max_length']]
y = df[['ROUGE-L', 'SacreBLEU', 'BLEU', 'BERTScore-F1']]

print("\nSplitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model (Random Forest) with progress tracking
print("\nTraining regression model...")
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Predict and evaluate with progress tracking
print("\nPredicting and evaluating the model...")
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")

# Print feature importances
importances = regressor.feature_importances_
print("\nFeature Importance:")
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")

