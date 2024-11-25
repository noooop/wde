import click

from wde.client import ChatClient
from wde.engine.zero_engine import start_zero_engine
from wde.tasks.chat.schema.api import ChatCompletionStreamResponseDone


@click.command()
@click.argument('model_name')
def run(model_name):
    engine_args = {"model": model_name}
    server = start_zero_engine(engine_args, bind_random_port=True)
    chat_client = ChatClient(nameserver_port=server.nameserver_port)

    print("正在加载模型...")
    chat_client.wait_service_available(model_name)
    print("加载完成!")
    print("!quit 退出, !next 开启新一轮对话。玩的开心！")

    try:
        quit = False
        while not quit:
            print("=" * 80)
            messages = []
            i = 1
            while True:
                print(f"[对话第{i}轮]")
                prompt = input("(用户输入:)\n")

                if prompt == "!quit":
                    quit = True
                    break

                if prompt == "!next":
                    break

                messages.append({"role": "user", "content": prompt})

                print(f"({model_name}:)\n", flush=True)
                content = ""
                for rep in chat_client.stream_chat(
                        model_name, messages, options={"max_tokens": 1024}):
                    if isinstance(rep, ChatCompletionStreamResponseDone):
                        print("\n", flush=True)
                        break
                    else:
                        print(rep.delta_content, end="", flush=True)
                        content += rep.delta_content

                messages.append({"role": "assistant", "content": content})
                i += 1
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        server.terminate()
        print("quit gracefully")


@click.group()
def chat():
    pass


chat.add_command(run)

if __name__ == '__main__':
    chat()
