import time

import click

from .make_dummy_inputs import make_dummy_inputs
from .speed_test import speed_test

@click.command()
@click.option('--model', default="THUDM/glm-4-9b-chat-1m")
@click.option("--length", default=100000)
@click.option("--filename", default="dummy.txt")
@click.option("--n", default=6)
def main(model: str, length: int, filename: str, n: int):
    for i in range(n):
        make_dummy_inputs(model, length, f"{i}-"+filename)

    for i in range(n):
        for j in range(i+1):
            speed_test(model, f"{j}-"+filename)

            time.sleep(1)

if __name__ == '__main__':
    main()