import torch
from transformers import BertTokenizer, BertForMaskedLM
from wordfreq import zipf_frequency


class Simplifier:

    def __init__(self):

        self.bert_model = 'bert-large-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.model = BertForMaskedLM.from_pretrained(self.bert_model)
        self.model.eval()

    def simplify_phrase(self, phrase, word):

        new_text = ""
        replace_word_mask = phrase.replace(word, '[MASK]')
        text = f'[CLS] {phrase} [SEP] {replace_word_mask} [SEP] '
        list_candidates_bert = []
        tokenized_text = self.tokenizer.tokenize(text)
        masked_index = [i for i, x in enumerate(tokenized_text) if x == '[MASK]'][0]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0][0][masked_index]
        predicted_ids = torch.argsort(predictions, descending=True)[:7]
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(list(predicted_ids))
        list_candidates_bert.append((word, predicted_tokens))

        for word_to_be_replaced, l_candidates in list_candidates_bert:
            tuples_word_zipf = []
            for w in l_candidates:
                if w.isalpha():
                    tuples_word_zipf.append((w, zipf_frequency(w, 'en')))
            tuples_word_zipf = sorted(tuples_word_zipf, key=lambda x: x[1], reverse=True)
            new_text = tuples_word_zipf[0][0]

        return new_text