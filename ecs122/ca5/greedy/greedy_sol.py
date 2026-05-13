from operator import itemgetter

def compatible(interval, interval_set):
    intervals = list(interval_set)

    # iterate over each interval already in the set
    for i in range(len(intervals)):
        if interval[0]

    return True


def main():
    intervals = [(6, 19),
                (20, 21),
                (19, 20),
                (20, 21),
                (18, 19),
                (8, 15),
                (15, 16),
                (7, 9),
                (18, 21),
                (16, 17)]
    
    # sort by finish times
    intervals.sort(key=itemgetter(1))
    print(intervals)

    # initialize empty set
    interval_set = set()

    for i in range(len(intervals)):
        if (compatible(intervals[i], interval_set)):
            set.add(intervals[i])

    return

if __name__ == "__main__":
    main()