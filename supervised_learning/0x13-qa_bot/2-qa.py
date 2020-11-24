#!/usr/bin/env python3
"""creates prompt to ask question"""
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf
words = ['exit', 'quit', 'goodbye', 'bye']


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
    if len(answer) <= 1 or answer == '[SEP]':
        return None
    return answer


def answer_loop(reference):
    """answers questions
    @reference is the referencce text
    Return answer"""

    while True:
        question = input("Q: ")

        if question.lower() in words:
            print('A: Goodbye')
            exit()
        else:
            answer = question_answer(question, reference)
            if answer is None:
                answer = "Sorry, I do not understand your question."
            print('A:', answer)
