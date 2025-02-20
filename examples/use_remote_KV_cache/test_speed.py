import time

import click
import requests


@click.command()
@click.option('--model', default="THUDM/glm-4-9b-chat-1m")
@click.option("--filename", default="dummy.txt")
def main(model, filename):
    with open(filename) as f:
        text = f.read()

    messages = [{
        'role': 'user',
        'content': f'请仔细阅读下面文字: {text[:50]}\n 概况主要内容：',
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


if __name__ == '__main__':
    main()
