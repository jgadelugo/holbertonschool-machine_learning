#!/usr/bin/env python3
"""creates prompt to ask question"""

if __name__ == "__main__":
    words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input("Q: ")

        if question.lower() in words:
            print('A: Goodbye')
            exit()
        else:
            print('A:')
