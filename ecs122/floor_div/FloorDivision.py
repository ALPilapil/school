def main():
    a, b = map(int, input().split())

    if b == 0:
        return "division by zero!!"
    else:
        return a // b

if __name__ == "__main__":
    result = main()
    print(result)