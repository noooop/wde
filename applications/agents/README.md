# 使用多智能体 agents

## 客户端支持多种 llm api
这个多智能体项目支持ollama和openai客户端，你可以任意选择熟悉的本地或网络api。

当然也可以试试本项目内部基于zmq的通讯协议。

## llm 服务端
不用费力的找了 llm 服务端了。本项目是个自带电池项目。

[配环境](https://github.com/noooop/wde/tree/main/setup)

项目自带与 ollama 和 openai 兼容的 webserver, 可以使用下面的部署文件一键部署

```commandline
wde serving applications/agents/deploy.yml
```

默认部署文件 “[deploy.yml](https://github.com/noooop/wde/tree/main/applications/agents/deploy.yml)” 部署 Qwen/Qwen2-7B-Instruct， 并部署 ollama 和 openai 网关。

## 先通过示例了解 agent 可以做啥

### Basic 
0. [多种协议支持 - 支持 ollama 和 openai 客户端、支持 zeroclient 内部通讯协议，支持流式推理](./notebook/t0_llm_client.ipynb)
1. [RolePlayingAgent - 角色扮演](./notebook/t1_role_playing_agent.ipynb)

### Multi-Agent
2. [Competitive - 对抗型](./notebook/t2_multi-agent-competitive.ipynb)
3. [Cooperative - 合作型](./notebook/t3_multi-agent-cooperative.ipynb)

### Workflow
4. [waterfall workflow - 多智能体顺序执行的工作流程](./notebook/t4_waterfall_workflow.ipynb)

### Conversation Patterns
5. [reflect - 反思，通过让大语言模型扮演多个角色，自问自答，不断提高输出结果](./notebook/t5_reflect.ipynb)
6. [divide and conquer - 分治，将一个大任务分成多个子任务，让大语言模型扮演多个角色分别完成子任务，汇总形成更全面的结果](./notebook/t6_divide_and_conquer.ipynb)

### Use tools
7. [use tools - 使用工具](./notebook/t7_use_tools.ipynb)


# 实现逻辑
1. 什么是Agent，如同人工智能一样，agent是不断发展快速演进的概念，恕我不在这里做出定义。理论上，Agent是这个项目实现能力的超集，共性存在于个性之中，所以Agent理论上应该是基类。
实际代码里 Agent = ConversableAgent。
2. ConversableAgent，我毫不怀疑很快大语言模型可以熟练使用文本图像语音甚至所有计算机支持的模态跟外界交流，但目前大语言模型只能用自然语言流畅的跟外界交流。
ConversableAgent 有 name 和 description 两个变量和 generate_reply 唯一外部接口，接受会话历史，产生回复。
ConversableAgent 抽象并不新奇，本质就是输入和输出都是string的、严格执行request-response的协议，所以http也是一种 ConversableAgent (狗头)。
本项目中 Agent 都没有状态，参考 serverless 可以叫它 agentless。
3. LLMAgent，是大语言模型接口的简单封装。在ConversableAgent基础上，变量增加system_message和llm_config，内部有chat_client与大语言模型推理引擎交互。
generate_reply 接受会话历史，调用chat_client与大语言模型推理引擎交互，产生回复。
很明显整个项目就是为了这点醋包的饺子。
4. RolePlayingAgent, 很快大家发现大语言模型不仅可以扮演 "You are a helpful AI Assistant." 扮演其他角色也活灵活现，大语言模型立即展现了巨大的潜力和商业价值。
在这个项目里，RolePlayingAgent = LLMAgent，你可以通过修改system_message赋予大语言模型不同人设，让大语言模型扮演不同角色
5. Multi-Agent, 多个具有不同人设的Agent可以进行交谈，把参与的agents拉入同一个Session， 由Session交替调用对应generate_reply，会话历史由Session保持。
多Agent系统根据任务类型，可以配置为对抗型(Competitive)和合作型(Cooperative)，这只是个开阔思路的分类学，并没有严格限定。可以发挥想象力创建非常复杂的场景，agent之间也有非常复杂的交互模式。
6. UserInput, 用户可以通过UserInput混迹于一群Agent中，你可以配置为其他Agent都围绕这User服务，也可以配置为User只是众多Agent中普通的一员。安利 2010年的电影《创：战纪 Tron: Legacy》希望给你一些启发.
7. Workflow, 内部复杂工作流，通过定义一个 agent 子类，封装复杂工作流在 generate_reply 完成。这里不区分代码写死的手工工作流、静态工作流，还是由大语言模型或方式实现自动选择分支的动态工作流，将工作流包装成agent方便测试和模块复用。
8. AgentUseTools，理论上你可以把任何输入和输出都是string的、严格执行request-response的协议的函数直接包装成Agent。但一般的函数都接受结构化输入输出，很难跟其他大语言模型或者人类使用自然语言直接交互。
所以要使用大语言模型包一层，非标转标，将自然语言转为函数需要的抽象格式化输入，函数输出整理为流畅具体的自然语言。 大语言模型还负责从提供的工具集中选择最适合的工具。
9. Agent Topology and Conversation Patterns 不同的交流模式适合解决不同的任务，逐渐发展出一些成熟的模式，比如reflect反思，divide and conquer分治...使用这些成熟的模式可以快速搭建复杂系统

# 设计理念
wde.agents 大幅借鉴了 [AutoGen](https://github.com/microsoft/autogen) 的设计理念，强烈建议去看看[原始论文](https://openreview.net/pdf?id=uAjxFFing2).

1. agents 只有一个外部接口 generate_reply 接受会话历史，产生回复
2. agents 自身没有记忆，就叫agentless吧，会话历史由Session保持
3. agents 自身没有“自动回复”等流程控制功能。agents之间的交替发言，必须把参与的agents拉入同一个Session， 由Session交替调用对应generate_reply
4. 内部复杂工作流，通过定义一个agent子类，封装复杂工作流在generate_reply完成。而不是使用nested chat，通过 register_reply 注册自定义回复函数
5. 使用gevent显示或隐式的并发加快运行， 个人非常讨厌 asyncio

AutoGen 复杂强大的 agent， wde.agents 简单弱小的 agent 都是语法糖，agent 的能力最终取决于底层大语言模型的能力。

# 更多 Agent 定义
很多 Agent 的论文会引用 Franklin and Graesser (1997) 对 Autonomous agents 的定义
> “An autonomous agent is a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future.”

很长时间，Autonomous agents 都是跟机器学习、深度学习、神经网络、自然语言处理等平行发展，直到ChatGPT时刻。
ChatGPT 的 in-Context Learning 和 follow instructions 的能力，以及chat自然语言交互方式深刻影响了包括 Autonomous agents 的众多领域。

以及 [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) 的定义和系统概述

> <img src="https://lilianweng.github.io/posts/2023-06-23-agent/agent-overview.png"  width="400">
> 
> 关键组成部分包括 Planning、Memory、Tool use、Action。

在 [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432) 中

> <img src="https://github.com/Paitesanshi/LLM-Agent-Survey/raw/main/assets/architecture-1.png?raw=true"  width="400">
>
> 关键组成部分包括 Profile、Memory、Planning、Action。

Agent 越来越变成大语言模型的样子。

Agent 原始的定义偏向于将其部署于未知的环境中，agent 提过观察周围环境，与周围环境交互，逐渐学习，最终完成既定目标。
而现在的大语言模型中心的应用，部署在完全为其发挥能力的环境中，环境对大语言模型都是已知的，尽可能按大语言模型设计、改造，手把手的教生怕大语言出问题。

说的更直白一点，一个用大语言模型做角色扮演的最简单应用，不涉及Planning、Memory、Tool use、Action，也没有观察周围环境，与周围环境交互，逐渐学习，是否可以称之为 Agent。
这不禁让人怀疑，大语言模型收编 agent 的过程，是否是对 “Agent” 这个词的滥用，而之前 “Agent” 指代的学科沦为大语言模型使用的一个场景。

回到 wde.agents 这个库来，Agent=ConversableAgent->LLMAgent=RolePlayingAgent->AgentUseTools，角色扮演称之为 Agent，AgentUseTools 沦为大语言模型使用的一个场景。
这样的抽象层次简单直观。虽然我对这样的分类学和 “Agent” 这个词的滥用有所保留。


