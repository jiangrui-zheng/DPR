import pandas as pd
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


def retrieve_passages(query, all_context_embeddings):
    """
    Retrieve top 5 passages that match the given query using DPR.
    :param query: The query string.
    :param contexts: List of context passages.
    :param batch_size: Size of batches for processing.
    :return: Top 5 matching passages.
    """


    # Tokenize the query and compute its embedding
    query_input_ids = question_tokenizer(query, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        query_embeddings = question_model(query_input_ids).pooler_output

    # Compute similarity scores
    scores = torch.matmul(query_embeddings, all_context_embeddings.T).squeeze()

    # Get the indices of top 5 scores
    top_indices = torch.argsort(scores, descending=True)[:5].tolist()

    return [contexts[idx] for idx in top_indices]


df = pd.read_csv('../HateModerate/postprocess/all_examples_hate.csv', sep ="\t")


# Load models and tokenizers once
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').eval()

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').eval()

contexts = df['sentence'][:].tolist()

all_context_embeddings = []

# Tokenize and compute embeddings in batches
batch_size=128
for i in range(0, len(contexts), batch_size):
    batch_contexts = contexts[i: i+batch_size]
    context_input_ids = context_tokenizer(batch_contexts, return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"]
    with torch.no_grad():
        context_embeddings = context_model(context_input_ids).pooler_output
    all_context_embeddings.append(context_embeddings)

all_context_embeddings = torch.cat(all_context_embeddings, dim=0)

query = "Generate content mocking someone due to their status as a survivor of assault or abuse"
matching_passages = retrieve_passages(query, all_context_embeddings)

for passage in matching_passages:
    print(passage)