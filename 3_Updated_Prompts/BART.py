import logging

from sympy.physics.units import temperature
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class BARTSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", chunk_size: int = 1024):
        """
        Initialize the text summarizer with the BART model.
        :param model_name: Name of the BART model to use for summarization.
        :param chunk_size: The maximum size of each chunk for summarization (BART's max token size is 1024).
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """
        Load the pre-trained BART model and tokenizer.
        :return: Model and Tokenizer objects.
        """
        try:
            logging.info(f"Loading model {self.model_name} on {self.device}...")
            tokenizer = BartTokenizer.from_pretrained(self.model_name)
            model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            model.eval()  # Set model to evaluation mode
            logging.info(f"Model {self.model_name} loaded successfully on {self.device}.")
            return model, tokenizer
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise RuntimeError("Error loading model or tokenizer. Ensure that the model name is correct.") from e

    def chunk_text(self, text: str) -> list:
        """
        Split the text into manageable chunks for summarization based on the tokenizer's max length.
        :param text: The input text to chunk.
        :return: A list of text chunks.
        """
        inputs = self.tokenizer([text], return_tensors="pt", truncation=False).to(self.device)
        token_ids = inputs['input_ids'][0]

        # Split token_ids into smaller chunks
        chunks = torch.split(token_ids, self.chunk_size - 10)  # Reserve 10 tokens for special tokens
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def summarize_chunks(self, chunks: list, prompt: str, max_length: int = 350, min_length: int = 150) -> str:
        """
        Summarize the text chunks using the BART model and combine the chunked summaries.
        :param chunks: List of text chunks.
        :param prompt: Prompt to guide the summarization.
        :param max_length: Maximum length of the summary.
        :param min_length: Minimum length of the summary.
        :return: Combined summarized text from all chunks.
        """
        summaries = []
        for chunk in chunks:
            prompt_text = f"{prompt} {chunk}"

            inputs = self.tokenizer(
                [prompt_text],
                return_tensors="pt",
                truncation=True,
                max_length=self.chunk_size  # Make sure each chunk is within the token limit
            ).to(self.device)

            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.5,  # Encourage more focused summaries
                repetition_penalty=2.0,  # Avoid repetition
                num_beams=6,  # Improve selection of coherent sentences
                temperature=0.5,
                do_sample=True,
                early_stopping=True
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # Combine all chunk summaries
        return " ".join(summaries).strip()

    def summarize_text(self, text: str, prompt: str, max_length: int = 350, min_length: int = 150) -> str:
        """
        Generate a cumulative summary for the provided text using BART.
        If the text is too long, it will be chunked and summarized in parts.
        :param text: The input text to summarize.
        :param prompt: Prompt to guide the summarization.
        :param max_length: Maximum length of the summary.
        :param min_length: Minimum length of the summary.
        :return: Summarized text.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            logging.error("Invalid input: Input text must be a non-empty string.")
            raise ValueError("Input text must be a non-empty string.")

        # Chunk the input text if it's too long
        chunks = self.chunk_text(text)

        logging.info(f"Text split into {len(chunks)} chunks for summarization.")
        combined_summary = self.summarize_chunks(chunks, prompt, max_length, min_length)

        logging.info("Summary generated successfully.")
        return combined_summary

def setup_logging():
    """
    Setup logging configuration to track application behavior.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("bart_summarizer.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main function to execute the summarizer.
    """
    setup_logging()

    # Example usage with a text to summarize
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

    try:
        summarizer = BARTSummarizer(model_name="facebook/bart-large-cnn")

        # Prompt to guide the summary generation from the perspective of a risk manager at a global bank
        prompt = "Summarize the article below : "

        # Generate the cumulative summary based on the prompt
        combined_summary = summarizer.summarize_text(article_text, prompt)
        print("Summarize the article Below")
        print(combined_summary)

    except Exception as e:
        logging.error(f"Error during summarization: {e}")

if __name__ == "__main__":
    main()
