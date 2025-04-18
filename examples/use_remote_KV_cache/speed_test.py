import time

import click
import requests


def speed_test(model, filename):
    with open(filename) as f:
        text = f.read()

    messages = [{
        'role': 'user',
        'content': f'请仔细阅读下面文字: {text}\n 概况主要内容：',
    }]

    start = time.perf_counter()

    response = requests.post('http://localhost:8080/v1/chat/completions',
                             json={
                                 "model": model,
                                 "messages": messages,
                                 "max_tokens": 1,
                                 "stream": False
                             })

    response.json()
    end = time.perf_counter()
    elapsed_time = end - start
    print("elapsed_time: ", elapsed_time)


@click.command()
@click.option('--model', default="THUDM/glm-4-9b-chat-1m-hf")
@click.option("--filename", default="dummy.txt")
def main(model, filename):
    speed_test(model, filename)


if __name__ == '__main__':
    main()
