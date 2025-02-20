import click

from wde.workflows.decoding.backends.sampling.utils import TokenSampler


@click.command()
@click.option('--model', default="THUDM/glm-4-9b-chat-1m")
@click.option("--length", default=9000)
@click.option("--filename", default="dummy.txt")
def main(model: str, length: int, filename: str):
    token_sampler = TokenSampler(model, trust_remote_code=True)
    text = token_sampler.random_sample(length, decode=True)
    with open(filename, "w") as f:
        f.write(text)


if __name__ == '__main__':
    main()
