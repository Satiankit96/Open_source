import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from tqdm import tqdm

class FinBERTSummarizer:
    def __init__(self, model_name="yiyanghkust/finbert-tone", summarizer_model="facebook/bart-large-cnn", max_input_length=512, device=None):
        """
        Initializes FinBERT for sentence extraction, BART for summarization, and FinBERT for sentiment analysis.
        Args:
            model_name: Pre-trained FinBERT model for financial text classification (default is FinBERT).
            summarizer_model: Pre-trained summarization model (default is BART-large).
            max_input_length: Maximum length for the input text (default 512 tokens).
            device: 'cuda' for GPU or 'cpu'. If not specified, it will automatically detect GPU availability.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load FinBERT for sentiment classification (financial sentiment extraction)
        self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        # Load BART for summarization
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model).to(self.device)
        self.max_input_length = max_input_length

    def preprocess_text(self, text):
        """
        Preprocesses the input text for summarization.
        Args:
            text: Input article text.
        Returns:
            Preprocessed text.
        """
        return text.strip().replace("\n", " ")

    def extract_key_sentences(self, text, threshold=0.5):
        """
        Uses FinBERT to extract sentences with positive or neutral sentiment above a given threshold.
        Args:
            text: The article to extract sentences from.
            threshold: The sentiment threshold (default 0.5) for extracting sentences.
        Returns:
            A list of key sentences based on sentiment analysis.
        """
        sentences = text.split('.')  # Simplistic sentence splitting (you can use better sentence tokenizers like `nltk`)

        key_sentences = []
        for sentence in sentences:
            inputs = self.finbert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            outputs = self.finbert_model(**inputs)
            sentiment = torch.softmax(outputs.logits, dim=1)  # Get sentiment scores

            # Check if the sentence has positive or neutral sentiment
            if sentiment[0][1].item() > threshold or sentiment[0][2].item() > threshold:  # Neutral or Positive
                key_sentences.append(sentence)

        return key_sentences

    def summarize(self, text, max_length=150, min_length=50, num_beams=5, prompt=""):
        """
        Summarizes the extracted key sentences and makes the prompt a key factor.
        Args:
            text: The article/document to summarize.
            max_length: Maximum length of the generated summary.
            min_length: Minimum length of the generated summary.
            num_beams: Beam search size for summary generation.
            prompt: A string prompt to guide the summarization.
        Returns:
            The generated summary.
        """
        # First, extract key sentences using FinBERT
        key_sentences = self.extract_key_sentences(text)
        if not key_sentences:
            return "No relevant sentences found based on sentiment."

        # Combine the key sentences and strongly integrate the prompt to guide the summary
        key_text = ' '.join(key_sentences)

        # Use the prompt as a strong prefix in multiple locations to enforce its influence
        prompt_text = f"Focus on the following key theme: {prompt}\n\n" + key_text  # Strong prompt influence

        # Tokenize and generate the summary
        inputs = self.summarizer_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_input_length).to(self.device)

        summary_ids = self.summarizer_model.generate(
            inputs["input_ids"],
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Perform sentiment analysis on the generated summary
        sentiment_analysis = self.analyze_sentiment(summary)

        return summary, sentiment_analysis

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of the given text using FinBERT.
        Args:
            text: The text to analyze (the generated summary).
        Returns:
            The sentiment analysis result: Positive, Neutral, or Negative.
        """
        inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        outputs = self.finbert_model(**inputs)
        sentiment = torch.softmax(outputs.logits, dim=1)  # Get sentiment scores

        sentiment_scores = sentiment[0].tolist()
        sentiment_labels = ["Negative", "Neutral", "Positive"]

        # Return the sentiment with the highest score
        sentiment_result = sentiment_labels[sentiment_scores.index(max(sentiment_scores))]

        return sentiment_result


# Example usage

class SummarizationPipeline:
    def __init__(self, finbert_model="yiyanghkust/finbert-tone", summarizer_model="facebook/bart-large-cnn", max_input_length=512, device=None):
        """
        Initializes the summarization pipeline with a FinBERTSummarizer object.
        Args:
            finbert_model: Pre-trained FinBERT model for financial text classification.
            summarizer_model: Pre-trained summarization model (default is BART-large).
            max_input_length: Maximum length for the input text (default 512 tokens).
            device: 'cuda' for GPU or 'cpu'. If not specified, it will automatically detect GPU availability.
        """
        self.summarizer = FinBERTSummarizer(finbert_model, summarizer_model, max_input_length, device)

    def run_single_summary(self, news_article, prompt=" Summarize potential cyber threats"):
        """
        Runs summarization on a single news article with a specified prompt and performs sentiment analysis on the summary.
        Args:
            news_article: Input news article text (string).
            prompt: The prompt to guide the summarization process.
        Returns:
            The generated summary and its sentiment analysis.
        """
        summary, sentiment = self.summarizer.summarize(news_article, prompt=prompt)
        print("Summary:", summary)
        print("Sentiment Analysis:", sentiment)


# Initialize the pipeline
pipeline = SummarizationPipeline()

# Example news article for summarization
news_article =  """
Abu Dhabi: Against the backdrop of a spike in cyber attacks globally, the UAE Cyber Security Council has warned users that criminals are using faster and more targeted techniques to hack users’ devices.
Speaking to Gulf News, Dr Mohamed Al Kuwaiti, the Council’s chairman, said that the Council had recently alerted users of several vulnerabilities in the operating systems of the devices of major international companies, allowing the attacker to control the devices.
The Council’s warning focused on protecting assets and sensitive information, pointing out the steps that must be followed to provide protection for all companies and government institutions.
In May, after the release of Fortinet’s latest Global Threat Landscape Report (which covered the second half of 2023), the Council recommended installing the latest version of software of a major company in order to avoid any hacks or leaks of personal information and data.
Dr Al Kuwaiti expects that ransomware attacks will continue to make headlines
The Report found that cyber attackers increased the speed at which they exploit vulnerabilities by 43 per cent, indicating a growing threat.
“Ransomware attacks will continue to make headlines, with the UAE and the broader Middle East being prime targets”, Dr Al Kuwaiti said. “Additionally, we are witnessing a rise in Distributed Denial of Service [DDoS] attack against UAE organisations, particularly against our critical infrastructure, amid a challenging geopolitical climate that amplifies cyber threats.”
He added: “It’s against this backdrop that the UAE Cyber Security Council is proud to introduce the State of the UAE Cybersecurity Report, which offers a comprehensive and accurate analysis of our nation’s cyber threat environment, providing actionable insights and recommendations. With cyber attacks on the rise, it is imperative the entire ecosystem to engage proactively in reducing the nation’s vulnerability to these threats.”
Last month, the Council had also called for the need to protect assets and enhance “physical digital security” to prevent hacks and financial losses, pointing out that weak physical security can lead to the leaking of companies’ confidential data, and legal responsibilities and financial repercussions for companies.
The Council explained that physical digital security is the protection of data and devices from physical actions that may cause loss or damage to digital assets.
Security measures
The Council identified five measures to enhance physical security:
• Be careful not to share sensitive information
• Ensure that unauthorised individuals cannot access private data
• Protect printed documents that contain sensitive information
• Track devices and erase data remotely if it is lost or stolen
• Use locked safes or secured storage solutions

Commenting on the Global Threat Landscape Report, Dr Al Kuwaiti said: “The global landscape of cyber threats indicates a worrying picture of the increasing speed in which cyber attackers exploit security vulnerabilities...”
He added: “The Report attributes this acceleration to multiple factors, including, firstly, artificial intelligence, as attackers employ artificial intelligence [AI] techniques to automate the processes of searching for and testing vulnerabilities, which enables them to exploit them much faster than human capabilities.
Profits from pirac
Dr Al Kuwait said another factor us is “profitable trade”, as electronic piracy operations generate huge profits for attackers, which motivates them to constantly develop their techniques and search for new vulnerabilities.
To address this rapid development in hacking technologies, Dr Al Kuwaiti called for automation and reducing burdens, as AI can automate many repetitive and arduous tasks in the field of cybersecurity, which reduces the burden on human analysts and allows them to focus on more complex tasks and continuous protection, as systems based on AI can work around the clock.
UAE takes action
“We would like to emphasise that the Cyber Security Council plays a pivotal role in combating these hacks,” Dr Al Kuwaiti said, listing measures such as training national cadres to deal with electronic attacks, enhancing the coordination between various cyber operations centres, raising awareness among community members about the importance of cyber security, and how to protect themselves from electronic attacks.
This message is confidential and subject to terms at: https://www.jpmorgan.com/emaildisclaimer including on confidential, privileged or legal entity information, malicious content and monitoring of electronic messages. If you are not the intended recipient, please delete this message and notify the sender immediately. Any unauthorized use is strictly prohibited.
"""

# Summarize a single article with the specified prompt and get sentiment analysis
pipeline.run_single_summary(news_article, prompt="Summarize the article below")
