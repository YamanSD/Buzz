from Data import get_data
from pandas import options
from Train import *


def main() -> None:
    """
    Main function of the program.
    :return:
    """
    # options.display.max_rows = 50
    # print(get_data(True))
    model, topic_mat, term_mat = train_lda(40_000, 10)
    # lsa_keys = get_keys(topic_mat)
    # lsa_cat, lsa_counts = keys_to_counts(lsa_keys)


    # top_n_words_lsa = get_top_n_words(8, 10, lsa_keys, term_mat, model)

    # for i in range(len(top_n_words_lsa)):
    #     print("Topic {}: ".format(i + 1), top_n_words_lsa[i])

    return


if __name__ == '__main__':
    main()
