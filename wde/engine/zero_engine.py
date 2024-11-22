import gc
import inspect

import torch

from wde import SamplingParams, const, envs
from wde.engine.gevent_engine import GeventLLMEngine
from wde.logger import init_logger
from wde.microservices.framework.zero.schema import (ZeroServerResponseOk,
                                                     ZeroServerStreamResponseOk
                                                     )
from wde.microservices.framework.zero.server import Z_MethodZeroServer
from wde.tasks.chat.schema.api import (ChatCompletionRequest,
                                       ChatCompletionResponse,
                                       ChatCompletionStreamResponse,
                                       ChatCompletionStreamResponseDone)
from wde.tasks.reranker.schema.api import RerankerRequest, RerankerResponse
from wde.tasks.retriever.schema.api import RetrieverRequest, RetrieverResponse
from wde.workflows.core.schema.engine_io import TextOnlyInputs

logger = init_logger(__name__)


class ZeroEngine(Z_MethodZeroServer):

    def __init__(self, name, engine_args, **kwargs):
        self.default_options = engine_args.pop("default_options", {})
        self.engine_args = engine_args
        self.engine = GeventLLMEngine(**self.engine_args)
        self.threadpool = self.engine.threadpool

        Z_MethodZeroServer.__init__(
            self,
            name=name,
            protocol=self.engine.engine.workflow.protocol,
            port=None,
            do_register=True,
            pool_size=self.engine.engine.engine_config.sys_config.
            zero_server_pool_size,
            **kwargs)

    def init(self):
        logger.info("%s %s is running! port: %d", self.__class__.__name__,
                    self.name, self.port)

    def z_info(self, req):
        rep = ZeroServerResponseOk(msg=self.engine.info)
        self.zero_send(req, rep)

    def __del__(self):
        try:
            self.engine.terminate()
            self.engine = None

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def get_metrics(self, output):
        if output.metrics is None:
            metrics = {}
        else:
            metrics = output.metrics.__dict__
        return metrics

    def z_encode(self, req):
        request = RetrieverRequest(**req.data)
        generator = self.engine.encode(inputs=request.inputs,
                                       request_id=str(req.req_id))

        output = list(generator)[0]
        metrics = self.get_metrics(output)

        response = RetrieverResponse(model=request.model,
                                     embedding=output.outputs.numpy(),
                                     metrics=metrics)

        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)

    def z_compute_score(self, req):
        request = RerankerRequest(**req.data)
        generator = self.engine.compute_score(inputs=request.pairs,
                                              request_id=str(req.req_id))

        output = list(generator)[0]
        metrics = self.get_metrics(output)

        response = RerankerResponse(model=request.model,
                                    score=output.score,
                                    metrics=metrics)

        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)

    def z_generate(self, req):
        request = ChatCompletionRequest(**req.data)
        assert self.engine.served_model_name == request.model

        options = request.options or {}
        options = {**options, **self.default_options}
        skip_empty_delta_text = options.pop("skip_empty_delta_text", True)
        request_id = str(req.req_id)
        sampling_params = SamplingParams(**options)

        def apply_chat_template(messages, tools=None):
            tokenizer = self.engine.get_tokenizer()

            if tools is None:
                prompt = tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    chat_template="tool_use",
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False)
            prompt_token_ids = tokenizer.encode(prompt)

            inputs = TextOnlyInputs(prompt=prompt,
                                    prompt_token_ids=prompt_token_ids)
            return inputs

        f = self.threadpool.submit(apply_chat_template, request.messages,
                                   request.tools)

        inputs = f.result()
        generator = self.engine.generate(inputs=inputs,
                                         request_id=request_id,
                                         sampling_params=sampling_params)

        def get_response():
            if not request.stream:
                final_res = None
                for res in generator:
                    final_res = res

                output = final_res.outputs[0]

                num_prompt_tokens = len(final_res.prompt_token_ids)
                num_generated_tokens = len(output.token_ids)

                return ChatCompletionResponse(
                    **{
                        "model": request.model,
                        "content": output.text,
                        "finish_reason": output.finish_reason,
                        "completion_tokens": num_generated_tokens,
                        "prompt_tokens": num_prompt_tokens,
                        "total_tokens": num_prompt_tokens +
                        num_generated_tokens
                    })

            else:

                def outputs_generator():
                    previous_texts = ""
                    prompt_tokens = 0
                    completion_tokens = 0
                    finish_reason = None

                    for res in generator:
                        output = res.outputs[0]

                        delta_text = output.text[len(previous_texts):]
                        previous_texts = output.text
                        finish_reason = output.finish_reason
                        prompt_tokens = len(res.prompt_token_ids)
                        completion_tokens = len(output.token_ids)

                        if not delta_text and skip_empty_delta_text:
                            continue

                        yield ChatCompletionStreamResponse(
                            **{
                                "model": request.model,
                                "delta_content": delta_text
                            })

                    yield ChatCompletionStreamResponseDone(
                        **{
                            "model": request.model,
                            "finish_reason": finish_reason,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        })

                return outputs_generator()

        response = get_response()

        if not inspect.isgenerator(response):
            rep = ZeroServerResponseOk(msg=response)
            self.zero_send(req, rep)
        else:
            for rep_id, rsp in enumerate(response):
                rep = ZeroServerStreamResponseOk(
                    msg=rsp,
                    snd_more=not isinstance(rsp,
                                            ChatCompletionStreamResponseDone),
                    rep_id=rep_id)
                self.zero_send(req, rep)


def start_zero_engine(engine_args):
    from wde.microservices.standalone.server import Server
    assert "model" in engine_args

    from wde.client import ZeroManagerClient

    server = Server()
    server.setup()
    server.run(waiting=False)

    manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
    manager_client.wait_service_available(envs.ROOT_MANAGER_NAME)

    model_name = engine_args["model"]

    manager_client.start(name=model_name,
                         engine_kwargs={
                             "server_class": const.INFERENCE_ENGINE_CLASS,
                             "engine_args": engine_args
                         })
    return server
