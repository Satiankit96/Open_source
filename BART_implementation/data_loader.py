def load_data():
    """
    Loads and returns the article text and human-generated summary.

    Returns:
        article_text (str): The text of the article.
        summary_text (str): The human-generated summary.
    """
    # Replace this with actual data loading logic, for example loading from a file or database
    article_text =  """
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

    summary_text ="""
    After a challenging few years, Chinese officials are steering activity and investment towards manufacturing and exports to compensate for domestic consumers’ reluctance to spend and a continuing crunch in the property market. As a result, China’s economy picked up pace in the first quarter, propelled by a rise in industrial production and swelling investment in factories. However this approach is leading to a lopsided recovery. Domestically, with familiar signs of weakness in consumption and real ­estate in the first three months of the year, many economists say Beijing still isn’t doing enough to support households and nurture a more balanced recovery. Outside China, Western governments and some big emerging economies are crying foul over a growing wave of cheap Chinese imports they say threaten domestic jobs and industries, while the International Monetary Fund and others warn that these mounting tensions over trade could lead to the global economy fracturing, with blocs of countries allied around the US and China, respectively.
    """
    return article_text, summary_text

def preprocess_text(text):
    """
    Preprocesses the input text by performing any necessary cleaning or tokenization.

    Args:
        text (str): The raw text to preprocess.

    Returns:
        preprocessed_text (str): The cleaned and tokenized text.
    """
    # Example preprocessing step (you can modify based on your needs)
    # Here we're simply trimming whitespaces, but you could also lower case, remove special characters, etc.
    preprocessed_text = text.strip()
    return preprocessed_text
