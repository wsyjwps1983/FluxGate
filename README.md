# FluxGate

[![FluxGate](https://i.ibb.co.com/g423grt/semantic-router-banner.png)](https://github.com/wsyjwps1983/FluxGate)

<p>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/semantic-router?logo=python&logoColor=gold" />
<a href="https://github.com/wsyjwps1983/FluxGate/graphs/contributors"><img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/wsyjwps1983/FluxGate" />
<a href="https://github.com/wsyjwps1983/FluxGate/commits/main"><img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/wsyjwps1983/FluxGate" />
<img alt="" src="https://img.shields.io/github/repo-size/wsyjwps1983/FluxGate" />
<a href="https://github.com/wsyjwps1983/FluxGate/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/wsyjwps1983/FluxGate" />
<a href="https://github.com/wsyjwps1983/FluxGate/pulls"><img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/wsyjwps1983/FluxGate" />
<a href="https://github.com/wsyjwps1983/FluxGate/blob/main/LICENSE"><img alt="Github License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

FluxGate is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — _routing_ our requests using _semantic_ meaning.

---

## Quickstart

To get started with FluxGate we install it like so:

```
pip install -qU semantic-router
```

❗️ _If wanting to use a fully local version of FluxGate you can use `LocalEncoder` (`pip install -qU "semantic-router[local]"`). To use the `HybridRouter` you must have the required dependencies installed._

We begin by defining a set of `Route` objects. These are the decision paths that the FluxGate can decide to use, let's try two simple routes for now — one for talk on _politics_ and another for _chitchat_:

```python
from semantic_router import Route

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

# we place both of our decisions together into single list
routes = [politics, chitchat]
```

## Encoders

FluxGate supports multiple encoders for generating embeddings, including support for SiliconFlow API. Here are some examples:

### SiliconFlow Encoder

```python
import os
from semantic_router.encoders import SiliconFlowEncoder

# Set SiliconFlow API key
os.environ["SILICONFLOW_API_KEY"] = "<YOUR_API_KEY>"

# Initialize SiliconFlowEncoder with a Chinese model
encoder = SiliconFlowEncoder(
    name="BAAI/bge-large-zh-v1.5",  # Chinese model
    score_threshold=0.7
)
```

### Other Encoders

```python
import os
from semantic_router.encoders import OpenAIEncoder, CohereEncoder, LocalEncoder

# OpenAI Encoder
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
openai_encoder = OpenAIEncoder()

# Cohere Encoder
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
cohere_encoder = CohereEncoder()

# Local Encoder (fully local)
local_encoder = LocalEncoder(name="all-MiniLM-L6-v2")
```

## Semantic Router

With our `routes` and `encoder` defined we now create a `SemanticRouter`. The router handles our semantic decision making.

```python
from semantic_router.routers import SemanticRouter

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

We can now use our router to make super fast decisions based on user queries:

```python
router("don't you love politics?").name
# Output: 'politics'

router("how's the weather today?").name
# Output: 'chitchat'

# Unrelated query returns None
result = router("I'm interested in learning about llama 2")
print(result)  # Output: None
```

## Dynamic Routes

Dynamic routes allow you to generate parameters and handle function calls dynamically. Here's an example:

```python
from semantic_router import Route
from semantic_router.routers import SemanticRouter

# Define a dynamic route for weather queries
weather = Route(
    name="weather",
    utterances=[
        "what's the weather like in {city}?",
        "how's the weather in {city} today?",
        "show me the forecast for {city}",
    ],
    function_call={
        "name": "get_weather",
        "description": "Get the current weather for a specific city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    }
)

routes = [weather]

# Initialize router with dynamic route
router = SemanticRouter(encoder=encoder, routes=routes)

# Test with a weather query
result = router("what's the weather like in Beijing?")
print(f"Route: {result.name}")
print(f"Function Call: {result.function_call}")
```

## Hybrid Router

Hybrid Router combines dense and sparse embeddings for improved performance. Here's an example:

```python
from semantic_router import Route
from semantic_router.routers import HybridRouter
from semantic_router.encoders import LocalEncoder, BM25Encoder
from semantic_router.index import FaissIndex

# Define routes
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
    ],
)

chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
    ],
)

routes = [politics, chitchat]

# Initialize encoders and index
encoder = LocalEncoder(name="all-MiniLM-L6-v2")
sparse_encoder = BM25Encoder()
index = FaissIndex()

# Initialize HybridRouter with alpha parameter (0.0 = sparse only, 1.0 = dense only)
router = HybridRouter(
    encoder=encoder,
    sparse_encoder=sparse_encoder,
    index=index,
    routes=routes,
    alpha=0.3  # Balance between dense and sparse embeddings
)

# Test the hybrid router
result = router("how's the weather in New York?")
print(f"Route: {result.name}")
print(f"Similarity Score: {result.similarity_score}")
```

## Integrations

FluxGate includes easy-to-use integrations with various services:

- **Encoders**: SiliconFlow, OpenAI, Cohere, Hugging Face, FastEmbed, Local Encoders
- **Vector Databases**: Pinecone, Qdrant, Faiss (local)
- **LLMs**: OpenAI, Mistral, Ollama, Cohere, and more
- **Frameworks**: LangChain, GraphAI

## 📚 Resources

### Documentation

- [API Usage Manual](docs/API_USAGE_MANUAL.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Threshold Optimization](docs/THRESHOLD_OPTIMIZATION.md)

### Examples

| Example | Description |
| -------- | ----------- |
| [Dynamic Routes](docs/02-dynamic-routes.ipynb) | Dynamic routes for parameter generation and function calls |
| [Local Execution](docs/05-local-execution.ipynb) | Fully local FluxGate with dynamic routes |
| [Hybrid Router](docs/examples/hybrid-router.ipynb) | Using HybridRouter for improved performance |
| [SiliconFlow Integration](build_router_with_siliconflow.py) | Example of using SiliconFlowEncoder |

---

## License

FluxGate is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.