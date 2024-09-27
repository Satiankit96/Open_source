import matplotlib.pyplot as plt
import pandas as pd

def plot_rouge_vs_parameters(results):
    """
    Plots ROUGE scores against different inference parameter combinations (num_beams, length_penalty, repetition_penalty).

    Args:
        results (DataFrame): DataFrame containing the tuning results with ROUGE scores.
    """
    num_beams = results['num_beams']
    rouge_scores = results['rouge_score']

    plt.figure(figsize=(8, 6))
    plt.plot(num_beams, rouge_scores, marker='o')
    plt.xlabel('num_beams')
    plt.ylabel('ROUGE-1 F1 Score')
    plt.title('ROUGE-1 F1 Score vs num_beams')
    plt.grid(True)
    plt.show()

def plot_bleu_vs_parameters(results):
    """
    Plots BLEU scores against different inference parameter combinations (num_beams, length_penalty, repetition_penalty).

    Args:
        results (DataFrame): DataFrame containing the tuning results with BLEU scores.
    """
    num_beams = results['num_beams']
    bleu_scores = results['bleu_score']

    plt.figure(figsize=(8, 6))
    plt.plot(num_beams, bleu_scores, marker='o')
    plt.xlabel('num_beams')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score vs num_beams')
    plt.grid(True)
    plt.show()

def plot_bert_vs_parameters(results):
    """
    Plots BERTScore F1 scores against different inference parameter combinations (num_beams, length_penalty, repetition_penalty).

    Args:
        results (DataFrame): DataFrame containing the tuning results with BERTScore F1 scores.
    """
    num_beams = results['num_beams']
    bert_f1_scores = results['bert_f1']

    plt.figure(figsize=(8, 6))
    plt.plot(num_beams, bert_f1_scores, marker='o')
    plt.xlabel('num_beams')
    plt.ylabel('BERTScore F1')
    plt.title('BERTScore F1 vs num_beams')
    plt.grid(True)
    plt.show()

def plot_metric_vs_length_penalty(results, metric='rouge'):
    """
    Plots the specified metric (ROUGE/BLEU/BERT) against the length_penalty parameter.

    Args:
        results (DataFrame): DataFrame containing the tuning results.
        metric (str): The metric to plot ('rouge', 'bleu', 'bert').
    """
    length_penalty = results['length_penalty']
    if metric == 'rouge':
        scores = results['rouge_score']
        ylabel = 'ROUGE-1 F1 Score'
    elif metric == 'bleu':
        scores = results['bleu_score']
        ylabel = 'BLEU Score'
    elif metric == 'bert':
        scores = results['bert_f1']
        ylabel = 'BERTScore F1'

    plt.figure(figsize=(8, 6))
    plt.plot(length_penalty, scores, marker='o')
    plt.xlabel('length_penalty')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs length_penalty')
    plt.grid(True)
    plt.show()

def plot_metric_vs_repetition_penalty(results, metric='rouge'):
    """
    Plots the specified metric (ROUGE/BLEU/BERT) against the repetition_penalty parameter.

    Args:
        results (DataFrame): DataFrame containing the tuning results.
        metric (str): The metric to plot ('rouge', 'bleu', 'bert').
    """
    repetition_penalty = results['repetition_penalty']
    if metric == 'rouge':
        scores = results['rouge_score']
        ylabel = 'ROUGE-1 F1 Score'
    elif metric == 'bleu':
        scores = results['bleu_score']
        ylabel = 'BLEU Score'
    elif metric == 'bert':
        scores = results['bert_f1']
        ylabel = 'BERTScore F1'

    plt.figure(figsize=(8, 6))
    plt.plot(repetition_penalty, scores, marker='o')
    plt.xlabel('repetition_penalty')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs repetition_penalty')
    plt.grid(True)
    plt.show()
