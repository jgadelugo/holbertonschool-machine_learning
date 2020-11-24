#!/usr/bin/env python3
"""Function that finds a snippet of text within a reference document
to answer a question
source1: https://tfhub.dev/see--/bert-uncased-tf2-qa/1
source2: https://huggingface.co/transformers/model_doc/bert.html
"""
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf


def question_answer(question, reference):
    """finds a snippet of text within a reference document
    to answer a question
    @question: string containing the question to answer
    @reference: string contatining the reference document from which to
    find the answer
    Return: string containing the answer
    """
    pre_trained = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(pre_trained)

    model = TFAutoModelForQuestionAnswering.from_pretrained(pre_trained,
                                                            return_dict=True)

    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + r_tokens + ['[SEP]']

    in_w_ids = tokenizer.convert_tokens_to_ids(tokens)
    in_mask = [1] * len(in_w_ids)
    in_t_ids = [0] * (1 + len(q_tokens) + 1) + [1] * (len(r_tokens) + 1)

    in_w_ids, in_mask, in_t_ids = map(lambda t: tf.expand_dims(
      tf.convert_to_tensor(t, dtype=tf.int32), 0), (in_w_ids,
                                                    in_mask, in_t_ids))
    outputs = model([in_w_ids, in_mask, in_t_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the
    # ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    # print(f'Question: {question}')
    # print(f'Answer: {answer}')
    return answer
