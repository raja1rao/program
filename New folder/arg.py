import argparse
def argm(args):
    result = None
    if args.operation == '+':
        result = args.num1 + args.num2
    if args.operation == '-':
        result = args.num1 - args.num2
    if args.operation == '*':
        result = args.num1 * args.num2
    if args.operation == 'pow':
        result = pow(args.num1, args.num2)

    print("Result : ", result)


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="my math script"
    )

    # Add the parameters positional/optional
    parser.add_argument('-n','--num1', help="Number 1", type=float)
    parser.add_argument('-i','--num2', help="Number 2", type=float)
    parser.add_argument('-o','--operation', help="provide operator", default='+')

    # Parse the arguments
    args = parser.parse_args()
    print(args)
    argm(args)
    