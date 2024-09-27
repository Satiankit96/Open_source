import logging
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from torch.utils.data import DataLoader, Dataset

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextSummarizer:
    def __init__(self, model_name: str = "google/pegasus-xsum"):
        """
        Initialize the text summarizer with a specific model.
        :param model_name: Name of the model to use for summarization.
        """
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.model.to(device)  # Move model to GPU if available

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

    def summarize_text(self, text: str, min_length: int = 50, max_length: int = 100) -> str:
        """
        Generate a summary for a single text.
        :param text: The input text to summarize.
        :param min_length: Minimum length of the summary.
        :param max_length: Maximum length of the summary.
        :return: Summarized text.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            logging.error("Invalid input: Input text must be a non-empty string.")
            raise ValueError("Input text must be a non-empty string.")

        # Tokenize and summarize the text
        logging.info("Tokenizing input text for summarization.")
        inputs = self.tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)

        logging.info("Generating summary...")
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=5,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.info("Summary generated successfully.")
        return summary


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

    # Example usage: single article summarization
    article_text = """
    China’s leaders have ambitious plans for the country’s economy, spanning one, five and even 15 years. In order to fulfil their goals, they know they will have to drum up prodigious amounts of manpower, materials and technology. But there is one vital input China’s leaders have recently struggled to procure: confidence.
According to the National Bureau of Statistics, consumer confidence collapsed in April 2022 when Shanghai and other big cities were locked down to fight the covid-19 pandemic (see chart 1). It has yet to recover. Indeed, confidence declined again in July, according to the latest survey. The figure is so bad it is a wonder the government still releases it.
Chart: The Economist
Gloom is not confined to consumers. Foreign companies have long complained about unfair or unpredictable policymaking. Some have declared China “uninvestible” as a consequence. Now their money is running along with their mouths. Foreign direct investment (FDI) in the country slumped to minus $14.8bn in the second quarter of this year, the worst figure on record. Any dollars ploughed in were comfortably outweighed by foreign investors selling stakes, collecting loan repayments or repatriating earnings. A separate figure that is calculated by the Ministry of Commerce dropped by almost 30% in yuan terms in the first seven months of this year, compared with the same period last year. Only during the global financial crisis of 2007-09 has FDI fallen as steeply.
More on this
Bad information is a grave threat to China’s economy
The Chinese authorities are concealing the state of the economy
Some of the blame lies elsewhere. Firms are reacting to trends outside the country as well as in it. America has, for example, discouraged investment in China’s semiconductor industry. And high American interest rates have lured money that might otherwise have stayed in the country.
But Chinese firms themselves are not much sunnier in their outlook. Each month government statisticians survey thousands of “purchasing managers” about their output, order books, hiring and outlook. According to the latest survey, business expectations fell in August to their lowest levels outside of the pandemic.
Although China’s leaders have resolved to “stabilise market expectations” and “enhance social confidence”, their proposed solutions are not terribly convincing. At a meeting of the Politburo in July, they urged cadres to “sing the bright future of China’s economy”. Expect FDI to fall further.
To lift the mood, officials must put their finger on its cause. Are people glum because the economy is bad? Or is the economy bad because people are glum?
A year ago it was possible to argue that corporate sentiment was merely a passive reflection of a weakening economy. Expectations were below their long-term average but so were new orders, according to the purchasing-managers’ indices. “Corporate confidence is still just a function of their order books,” as Christopher Beddor and Thomas Gatley of Gavekal Dragonomics, a consultancy, put it in August 2023. “The best way to improve expectations and investment behaviour is simply to improve current economic conditions through more stimulus,” they concluded. That argument is now harder to make. Sentiment has deteriorated over the past year even faster than new orders (see chart 2). Expectations are now worse than you would expect given other indicators of activity.
Chart: The Economist
Exports have, for example, held up surprising well so far this year. It is nevertheless telling that Chinese firms have been slow to convert their foreign earnings into China’s currency. Over the past two years they have clung on to about $400bn they would typically have converted into yuan, according to estimates by Goldman Sachs, a bank. Now that American interest rates are likely to fall, the incentive to hold dollars should diminish. But unless the future of China’s economy brightens, exporters may not rush to acquire yuan instead.
Some analysts think that China’s gloom reflects deeper problems, beyond current economic circumstances. Adam Posen of the Peterson Institute for International Economics, a think-tank, has argued that faith in China’s policymaking was shattered by the pandemic lockdowns, as well as by abrupt regulatory crackdowns on some of China’s most celebrated companies. In both cases, officials cast private prosperity aside in pursuit of other goals.
China’s leaders have tried to regain the trust of entrepreneurs. Policymakers are drafting a private-sector promotion law. But if the party’s mood were to turn hostile again, new laws would offer little protection. China’s ruling party cannot convincingly restrain itself: it lacks the power to limit its own power. As such, private insecurity may blunt the impact of government stimulus, Mr Posen believes. Low interest rates will not tempt people to borrow and spend. Increased government expenditure will not crowd in private spending.
The two rival explanations for China’s confidence problem thus have quite different implications for policy. If gloom is simply a function of a weak economy, stronger stimulus should dispel it. Conversely, if Mr Posen is right, stimulus will revive neither the economy nor confidence.
At various points over the past year China’s government has seemed poised to settle the argument. In order to stimulate the economy, it has signalled an easing of fiscal policy. It has authorised repeated sales of long-term government bonds and approved a sizeable quota of “special bonds” that local governments could sell themselves. China’s central bank has also offered cheap financing to help stabilise the property market.
But the country’s budget deficit, broadly defined, actually shrank in the first half of the year, according to Goldman Sachs. Little of the central bank’s financing has been tapped. And local governments were surprisingly slow to issue bonds, even as other sources of revenue, such as land sales, dried up faster than assumed in budget documents. “Local governments are becoming more and more passive,” complained Zhao Jian of Xijing Research Institute, in a report that was quickly censored by the authorities.
Why this sluggishness? Local officials may have been slow to tackle weak morale because they are suffering something of a confidence crisis of their own. In order to curb corruption and ensure Beijing’s priorities are faithfully implemented, Xi Jinping, China’s leader, has curtailed local discretion and subjected cadres to closer scrutiny. Lower-level officials now have less authority but more accountability, points out Jessica Teets of Middlebury College. Their power has declined. Their risk of punishment has increased.
The number of cases filed by the Central Commission for Discipline Inspection, China’s graft-buster, rose by 28% in the first half of this year, compared with a year earlier. Officials are working longer hours and spending more time filling out forms. In this context, local policymakers are reluctant to make any bold displays of initiative. They reason that “the more you do, the more mistakes you will make”, according to Ms Teets, who surveyed officials about such matters in 2022. Beijing might urge stimulus spending today. But it is also possible it will criticise the resulting debts or choice of projects in the future.
Ms Teets found that a third of local officials would quit if they had the chance, so miserable were they in their jobs. To restore the private sector’s faith in policymaking, China must first restore the morale of its policymakers. Perhaps the Politburo can lead them in song.
"""

    try:
        summarizer = TextSummarizer(model_name="google/pegasus-xsum")
        summary = summarizer.summarize_text(article_text)
        print("Generated Summary:")
        print(summary)

    except Exception as e:
        logging.error(f"Error during summarization: {e}")


if __name__ == "__main__":
    main()
