import os
from asyncio import sleep as asleep
from time import sleep
from typing import Any, List, Optional, Union

import openai
import tiktoken
from openai import OpenAIError
from openai._types import NotGiven
from openai.types import CreateEmbeddingResponse
from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder
from semantic_router.schema import EncoderInfo
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger

# SiliconFlow支持的模型配置
siliconflow_model_configs = {
    "BAAI/bge-large-zh-v1.5": EncoderInfo(
        name="BAAI/bge-large-zh-v1.5",
        token_limit=512,
        threshold=0.7,
    ),
    "BAAI/bge-base-zh-v1.5": EncoderInfo(
        name="BAAI/bge-base-zh-v1.5",
        token_limit=512,
        threshold=0.7,
    ),
    "BAAI/bge-small-zh-v1.5": EncoderInfo(
        name="BAAI/bge-small-zh-v1.5",
        token_limit=512,
        threshold=0.7,
    ),
    "BAAI/bge-large-en-v1.5": EncoderInfo(
        name="BAAI/bge-large-en-v1.5",
        token_limit=512,
        threshold=0.7,
    ),
    "text-embedding-ada-002": EncoderInfo(
        name="text-embedding-ada-002",
        token_limit=8192,
        threshold=0.82,
    ),
    "text-embedding-3-small": EncoderInfo(
        name="text-embedding-3-small",
        token_limit=8192,
        threshold=0.3,
    ),
    "text-embedding-3-large": EncoderInfo(
        name="text-embedding-3-large",
        token_limit=8192,
        threshold=0.3,
    ),
}


class SiliconFlowEncoder(DenseEncoder):
    """SiliconFlow encoder class for generating embeddings using SiliconFlow API.

    The SiliconFlowEncoder class is a subclass of DenseEncoder and utilizes the SiliconFlow API
    to generate embeddings for given documents. It requires a SiliconFlow API key and
    supports customization of the score threshold for filtering or processing the embeddings.
    """

    _client: Optional[openai.Client] = PrivateAttr(default=None)
    _async_client: Optional[openai.AsyncClient] = PrivateAttr(default=None)
    dimensions: Union[int, NotGiven] = NotGiven()
    token_limit: int = 8192  # default value, should be replaced by config
    _token_encoder: Any = PrivateAttr()
    type: str = "siliconflow"
    max_retries: int = 3

    def __init__(
        self,
        name: Optional[str] = None,
        siliconflow_api_key: Optional[str] = None,
        siliconflow_base_url: Optional[str] = None,
        score_threshold: Optional[float] = None,
        dimensions: Union[int, NotGiven] = NotGiven(),
        max_retries: int = 3,
    ):
        """Initialize the SiliconFlowEncoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param siliconflow_api_key: The SiliconFlow API key.
        :type siliconflow_api_key: str
        :param siliconflow_base_url: The base URL for the SiliconFlow API.
        :type siliconflow_base_url: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        :param dimensions: The dimensions of the embeddings.
        :type dimensions: int
        :param max_retries: The maximum number of retries for the SiliconFlow API call.
        :type max_retries: int
        """
        if name is None:
            name = "BAAI/bge-large-zh-v1.5"  # 默认使用中文模型
            
        if score_threshold is None and name in siliconflow_model_configs:
            set_score_threshold = siliconflow_model_configs[name].threshold
        elif score_threshold is None:
            logger.warning(
                f"Score threshold not set for model: {name}. Using default value."
            )
            set_score_threshold = 0.7
        else:
            set_score_threshold = score_threshold
            
        super().__init__(
            name=name,
            score_threshold=set_score_threshold,
        )
        
        api_key = siliconflow_api_key or os.getenv("SILICONFLOW_API_KEY")
        base_url = siliconflow_base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        
        if api_key is None or api_key.strip() == "":
            raise ValueError("SiliconFlow API key cannot be 'None' or empty.")
            
        if max_retries is not None:
            self.max_retries = max_retries
            
        try:
            self._client = openai.Client(
                base_url=base_url, api_key=api_key
            )
            self._async_client = openai.AsyncClient(
                base_url=base_url, api_key=api_key
            )
        except Exception as e:
            raise ValueError(
                f"SiliconFlow API client failed to initialize. Error: {e}"
            ) from e
            
        # set dimensions to support embed 3 dimensions param
        self.dimensions = dimensions
        
        # if model name is known, set token limit
        if name in siliconflow_model_configs:
            self.token_limit = siliconflow_model_configs[name].token_limit
        else:
            # 尝试使用OpenAI的tokenizer作为备选
            try:
                self._token_encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
            except:
                self._token_encoder = tiktoken.get_encoding("cl100k_base")

    def __call__(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using SiliconFlow API.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self._client is None:
            raise ValueError("SiliconFlow client is not initialized.")
        
        if truncate:
            # check if any document exceeds token limit and truncate if so
            docs = [self._truncate(doc) for doc in docs]

        # SiliconFlow API has a batch size limit of 32, so we need to process in batches
        batch_size = 30  # Use 30 to be safe (max is 32)
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            embeds = None
            
            # Exponential backoff for each batch
            for j in range(self.max_retries + 1):
                try:
                    logger.debug(f"Creating embeddings for batch {i//batch_size + 1}, {len(batch)} docs")
                    embeds = self._client.embeddings.create(
                        input=batch,
                        model=self.name,
                        dimensions=self.dimensions,  # type: ignore[arg-type]
                    )
                    if embeds.data:
                        break
                except OpenAIError as e:
                    logger.error("Exception occurred", exc_info=True)
                    if self.max_retries != 0 and j < self.max_retries:
                        sleep(2**j)
                        logger.warning(
                            f"Retrying in {2**j} seconds due to OpenAIError: {e}"
                        )
                    else:
                        raise

                except Exception as e:
                    logger.error(f"SiliconFlow API call failed. Error: {e}")
                    raise ValueError(f"SiliconFlow API call failed. Error: {str(e)}") from e

            if (
                not embeds
                or not isinstance(embeds, CreateEmbeddingResponse)
                or not embeds.data
            ):
                logger.info(f"Returned embeddings: {embeds}")
                raise ValueError("No embeddings returned.")

            batch_embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
            all_embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to avoid rate limiting
            if i + batch_size < len(docs):
                sleep(0.5)

        return all_embeddings

    def _truncate(self, text: str) -> str:
        """Truncate a document to the token limit.

        :param text: The document to truncate.
        :type text: str
        :return: The truncated document.
        :rtype: str
        """
        # 使用tiktoken来估算token数量
        if not hasattr(self, '_token_encoder'):
            return text
            
        # we use encode_ordinary as faster equivalent to encode(text, disallowed_special=())
        tokens = self._token_encoder.encode_ordinary(text)
        if len(tokens) > self.token_limit:
            logger.warning(
                f"Document exceeds token limit: {len(tokens)} > {self.token_limit}"
                "\nTruncating document..."
            )
            text = self._token_encoder.decode(tokens[: self.token_limit - 1])
            logger.info(f"Trunc length: {len(self._token_encoder.encode(text))}")
            return text
        return text

    async def acall(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using SiliconFlow API asynchronously.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self._async_client is None:
            raise ValueError("SiliconFlow async client is not initialized.")
        
        if truncate:
            # check if any document exceeds token limit and truncate if so
            docs = [self._truncate(doc) for doc in docs]

        # SiliconFlow API has a batch size limit of 32, so we need to process in batches
        batch_size = 30  # Use 30 to be safe (max is 32)
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            embeds = None
            
            # Exponential backoff for each batch
            for j in range(self.max_retries + 1):
                try:
                    logger.debug(f"Creating embeddings for batch {i//batch_size + 1}, {len(batch)} docs")
                    embeds = await self._async_client.embeddings.create(
                        input=batch,
                        model=self.name,
                        dimensions=self.dimensions,  # type: ignore[arg-type]
                    )
                    if embeds.data:
                        break
                except OpenAIError as e:
                    logger.error("Exception occurred", exc_info=True)
                    if self.max_retries != 0 and j < self.max_retries:
                        await asleep(2**j)
                        logger.warning(
                            f"Retrying in {2**j} seconds due to OpenAIError: {e}"
                        )
                    else:
                        raise

                except Exception as e:
                    logger.error(f"SiliconFlow API call failed. Error: {e}")
                    raise ValueError(f"SiliconFlow API call failed. Error: {e}") from e

            if (
                not embeds
                or not isinstance(embeds, CreateEmbeddingResponse)
                or not embeds.data
            ):
                logger.info(f"Returned embeddings: {embeds}")
                raise ValueError("No embeddings returned.")

            batch_embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
            all_embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to avoid rate limiting
            if i + batch_size < len(docs):
                await asleep(0.5)

        return all_embeddings