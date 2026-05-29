import sys

def detect_match_util(text_string, pattern_string, text_idx, pattern_idx):
    '''
    main workhorse of pattern matching  
    '''
    # base case 
    # both characters are null 


    
    # shrink problem    
    
    # call recursion 

def detect_match(text_string, pattern_string):
    '''
    detect macthes based on these two text_string
    call a recursive util function 
    '''
    # * = blank or any amount
    # . = one 
    # + = non empty string 
    # ? = empty or one 
    
    


def main():
    """
    
    """
    # get input
    lines = sys.stdin.read().splitlines()
    num_cases = int(lines[0])
    
    # evaluate for every pair 
    for i in range(1, len(lines), 2):
        text_string = lines[i]
        pattern_string = lines[i + 1]

        match_bool = detect_match(text_string, pattern_string)

    return 


if __name__ == "__main__":
    main()
