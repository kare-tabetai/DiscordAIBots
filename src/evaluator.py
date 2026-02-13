"""Image evaluation using Ollama vision models."""

import asyncio
import base64

import ollama


class ImageEvaluator:
    """Evaluates images using Ollama vision models."""

    def __init__(self, host: str, model: str, default_prompt: str):
        """Initialize the image evaluator.

        Args:
            host: Ollama server host URL.
            model: Vision model name (e.g., 'minicpm-v').
            default_prompt: Default prompt for image evaluation.
        """
        self.host = host
        self.model = model
        self.default_prompt = default_prompt
        self._client = ollama.Client(host=host)

    def _evaluate_sync(self, image_data: bytes, prompt: str | None = None) -> str:
        """Synchronously evaluate an image.

        Args:
            image_data: Image binary data.
            prompt: Custom prompt for evaluation. Uses default if None.

        Returns:
            Evaluation result text.
        """
        eval_prompt = prompt or self.default_prompt
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        response = self._client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": eval_prompt,
                    "images": [image_base64],
                }
            ],
        )

        return response["message"]["content"]

    async def evaluate(self, image_data: bytes, prompt: str | None = None) -> str:
        """Asynchronously evaluate an image.

        Args:
            image_data: Image binary data.
            prompt: Custom prompt for evaluation. Uses default if None.

        Returns:
            Evaluation result text.
        """
        return await asyncio.to_thread(self._evaluate_sync, image_data, prompt)
