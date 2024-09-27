import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import logging

nltk.download('punkt')

class PegasusSummarizer:
    def __init__(self, model_name: str = "google/pegasus-xsum", max_length: int = 1024, min_length: int = 50):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.max_length = max_length
        self.min_length = min_length

    def load_model_and_tokenizer(self):
        try:
            logging.info(f"Loading model {self.model_name} on {self.device}...")
            tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            model.eval()
            logging.info(f"Model {self.model_name} loaded successfully on {self.device}.")
            return model, tokenizer
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise RuntimeError("Error loading model or tokenizer.") from e

    def preprocess_text(self, text: str):
        sentences = sent_tokenize(text)
        cleaned_text = ' '.join([sentence.strip() for sentence in sentences if sentence.strip()])
        return cleaned_text

    def summarize_text(self, text: str, num_beams: int = 4, repetition_penalty: float = 1.2):
        preprocessed_text = self.preprocess_text(text)

        if not preprocessed_text or len(preprocessed_text.split()) < 5:
            logging.warning(f"Text is too short or empty: {preprocessed_text}")
            return "Text too short for summarization."

        try:
            inputs = self.tokenizer(
                [preprocessed_text],
                truncation=True,
                padding='longest',
                return_tensors="pt",
                max_length=self.max_length
            ).to(self.device)

            if inputs['input_ids'].shape[1] == 0:
                logging.error("Tokenization resulted in empty input IDs. Skipping summarization.")
                return "Error during summarization."

            summary_ids = self.model.generate(
                inputs['input_ids'],
                num_beams=num_beams,
                max_length=self.max_length,
                min_length=self.min_length,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return "Error during summarization."

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("pegasus_summarizer.log"),
            logging.StreamHandler()
        ]
    )

def summarize_article_batch(article_texts, num_beams=4, max_length=1024, min_length=50):
    summarizer = PegasusSummarizer(max_length=max_length, min_length=min_length)
    summaries = []
    for article_text in tqdm(article_texts, desc="Summarizing Articles"):
        summary = summarizer.summarize_text(article_text, num_beams=num_beams)
        summaries.append(summary)
    return summaries

def main():
    setup_logging()
    articles = [
        """
            The headquarters of Russia’s military intelligence agency in Moscow. 
        In January, an alert citizen in Muleshoe, Tex., was driving by a park and noticed that a water tower was overflowing. Authorities soon determined the system that controlled the city’s water supply had been hacked. In two hours, tens of thousands of gallons of water had flowed into the street and drain pipes.
        The hackers posted a video online of the town’s water-control systems and a nearby town being manipulated, showing how they reset the controls. In the video on the messaging platform Telegram, they called themselves Cyber Army of Russia Reborn (CARR).
        “We’re starting another raid on the USA,” the video caption reads in Russian, with the hackers saying they would show how they exploited “a couple critical infrastructure facilities, namely water supply systems.” It was followed by a smiley face emoji.
        That water tank overflow in a Texas panhandle town may well be linked to one of the most infamous Russian government hacking groups, the cybersecurity firm Mandiant said Wednesday. 
        If confirmed, analysts say it would mark a worrisome escalation by Moscow in its attempts to disrupt critical U.S. infrastructure by targeting one of its weakest sectors: water utilities. 
        The hacking group, which private sector analysts once dubbed Sandworm, has achieved notoriety for briefly turning out the lights in parts of Ukraine at least three different times; hacking the Olympics Opening Games in South Korea in 2018; and launching NotPetya, one of the most damaging cyberattacks ever that cost businesses worldwide tens of billions of dollars.
        Although no one was hurt and service was not interrupted in Muleshoe, the prospect of Sandworm broadening its sites from Ukrainian power grids and French elections to American critical infrastructure is troubling, Mandiant chief analyst John Hultquist said. 
        The U.S. government assesses Sandworm to be part of the GRU, Russia’s military spy agency.
        The team at Mandiant, which is owned by Google, observed social media accounts being created on YouTube for CARR using servers associated with Sandworm, Hultquist said, adding that Mandiant also has found CARR posting Ukrainian government data stolen by Sandworm hackers on Telegram.
        “We’ve been saying for a long time that CARR is just a front for the GRU,” Hultquist said. “Then we see them take credit for these acts in the U.S. against water utilities. Is GRU behind these attacks? If it isn’t GRU, whoever is doing this is working out of the same clubhouse. It’s too close for comfort.”
        The U.S. intelligence community has not yet made a determination whether CARR is run by the GRU, although intelligence analysts are scouring clues.
        Robert M. Lee, CEO and co-founder of Dragos, which specializes in industrial control system cybersecurity, said a team from his firm tracked CARR’s operations in January. He confirmed the water overflow in Muleshoe but could not specify whether this happened in other towns. “The adversary was definitely looking to do disruptions,” he said, noting that the trend over the last several years has been for state actors to seek to disrupt systems, whereas a decade ago, they were interested mostly in espionage.
        Another target was the nearby town of Abernathy. The city’s manager, Don Provost, said in an interview that the hack “didn’t interrupt anything.” The FBI and Department of Homeland Security got in touch quickly, he said.
        “It actually turned out to be a good thing,” he said. “It showed us where our vulnerabilities were.”
        In an interview, Muleshoe’s city manager, Ramon Sanchez, said the hackers brute-forced the password for the system’s control system interface, which was run by a vendor. That password hadn’t been changed in more than a decade, he admitted. 
        “You don’t think that’s going to happen to you. It’s always going to happen to the other guy,” he said.
        The same vendor was used by at least two other towns in the area that were subjected to attempted hacks, Sanchez said.
        But the incident also forced changes. “We learned,” Sanchez said. “The biggest lesson is that we have to always be proactive and always update our cybersecurity.”
        He thinks Muleshoe was a “victim of opportunity,” adding: “I would have never thought that somebody tied to the Russian military would target Muleshoe.”
            """
    ]

    try:
        logging.info("Starting batch summarization...")
        summaries = summarize_article_batch(articles, num_beams=4, max_length=1024, min_length=50)
        logging.info("Summarization completed.")

        for i, summary in enumerate(summaries):
            print(f"Summary {i+1}:")
            print(summary)
            print("\n")

    except Exception as e:
        logging.error(f"Error during summarization: {e}")

if __name__ == "__main__":
    main()
