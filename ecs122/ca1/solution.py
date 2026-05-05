def main():
    # get input
    length, radius = map(int, input().split())
    input_list = sorted(map(int, input().split()))

    # check coverage for each city
    left_idx = 0
    right_idx = 0
    for i in range(length):
        # while the left_idx location is less than the current location - radius, move left_idx up
        while input_list[left_idx] < input_list[i] - radius:
            left_idx += 1
        # while the right_idx location + 1 is less than the length, avoiding idx error, and is less than or equal to the current + radius, expand further right_idx
        while right_idx + 1 < length and input_list[right_idx + 1] <= input_list[i] + radius:
            right_idx += 1
        
        # right_idx is how far ahead we are, left_idx is how far back, therefore the total coverage is right_idx - left_idx
        print(right_idx - left_idx + 1)

    # find max coverage
    best = 0
    right_idx = 0
    for i in range(length):
        # same logic as before, just moving right_idx now according to 2 * radius because left_idx most is set 
        while right_idx + 1 < length and input_list[right_idx + 1] <= input_list[i] + 2 * radius:
            right_idx += 1
        count = right_idx - i + 1
        # update 
        if count > best:
            best = count

    print(best)

    return 0

main()