import sys

def main():
    names = sys.stdin.read().split()

    for name in names:
        print(f"Hello {name}!")

    return 0

if __name__ == "__main__":
    main()