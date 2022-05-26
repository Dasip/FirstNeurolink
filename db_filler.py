file = "words.txt"

CURRENT_WORD = ''
CURRENT_FILE = None


def process_word():
    global CURRENT_WORD
    print('Введите слово:')
    word = input().lower()
    if word != '0':
        CURRENT_WORD = word
        process_weight()
    else:
        CURRENT_FILE.close()


def process_weight():
    print('Введите вес:')
    weight = str(float(input()))
    CURRENT_FILE.write(f'{CURRENT_WORD}:{weight}\n')
    process_word()


if __name__ == "__main__":
    CURRENT_FILE = open(file, 'a')
    process_word()