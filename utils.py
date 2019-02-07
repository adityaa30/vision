def print_list(list):
    """
    Displays each element of the list in separate line
    :param list: 1D list containing elements
    """
    output_string = ''
    for item in list:
        output_string += str(item) + '\n'

    return output_string


def mark_captions(captions_list, mark_start='<SOS> ', mark_end=' <EOS>'):
    """
    Wraps all text-strings in the start[*]end markers.

    :param captions_list: list of lists with text-captions
    :param mark_start: starting marker for each caption
    :param mark_end: ending marker for each caption
    """
    captions_marked = [
        [
            mark_start + caption + mark_end
            for caption in captions
        ]
        for captions in captions_list
    ]

    return captions_marked


def flatten_captions(captions_list):
    """
    Converts a list-of-list to a flattened list of captions.

    :param captions_list: list of lists with text-captions
    """
    captions_list = [
        caption
        for captions in captions_list
        for caption in captions
    ]

    return captions_list
