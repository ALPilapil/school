import sys

def detect_match_util(text_string, pattern_string, text_idx, pattern_idx):
    '''
    main workhorse of pattern matching
    '''
    # base case
    # both characters are null
    if text_idx == len(text_string) and pattern_idx == len(pattern_string):
        return True

    # pattern exhausted, text remains
    if pattern_idx == len(pattern_string):
        return False

    # text exhausted, pattern remains — only * and ? can match empty
    if text_idx == len(text_string):
        if pattern_string[pattern_idx] in ('*', '?'):
            return detect_match_util(text_string, pattern_string, text_idx, pattern_idx + 1)
        return False

    # shrink problem
    c = pattern_string[pattern_idx]

    # call recursion
    if c == '*':
        # skip * (match empty) OR consume one char and keep *
        return (detect_match_util(text_string, pattern_string, text_idx, pattern_idx + 1) or
                detect_match_util(text_string, pattern_string, text_idx + 1, pattern_idx))
    elif c == '.':
        # consume exactly one char
        return detect_match_util(text_string, pattern_string, text_idx + 1, pattern_idx + 1)
    elif c == '+':
        # consume one (required), then keep + (more chars) OR advance pattern
        return (detect_match_util(text_string, pattern_string, text_idx + 1, pattern_idx) or
                detect_match_util(text_string, pattern_string, text_idx + 1, pattern_idx + 1))
    elif c == '?':
        # skip (match empty) OR consume one char
        return (detect_match_util(text_string, pattern_string, text_idx, pattern_idx + 1) or
                detect_match_util(text_string, pattern_string, text_idx + 1, pattern_idx + 1))
    else:
        # literal: must match exactly
        if text_string[text_idx] == c:
            return detect_match_util(text_string, pattern_string, text_idx + 1, pattern_idx + 1)
        return False

def detect_match(text_string, pattern_string):
    '''
    detect macthes based on these two text_string
    call a recursive util function
    '''
    # * = blank or any amount
    # . = one
    # + = non empty string
    # ? = empty or one
    return detect_match_util(text_string, pattern_string, 0, 0)


def main():
    """
    get input and process each pair one at a time, outputting
    the result for each one right away  
    """
    # get input
    lines = sys.stdin.read().splitlines()
    num_cases = int(lines[0])

    # evaluate for every pair
    for i in range(1, len(lines), 2):
        text_string = lines[i]
        pattern_string = lines[i + 1]

        match_bool = detect_match(text_string, pattern_string)

        # should be text
        print("true" if match_bool else "false")

    return


if __name__ == "__main__":
    main()
