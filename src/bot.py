"""Discord Bot for image evaluation."""

import asyncio
import logging

import discord

from .config import Config
from .evaluator import ImageEvaluator
from .gdrive import GoogleDriveClient

logger = logging.getLogger(__name__)


class ImageEvaluationBot(discord.Client):
    """Discord Bot that evaluates images from Google Drive."""

    def __init__(self, config: Config):
        """Initialize the bot.

        Args:
            config: Application configuration.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config
        self.gdrive = GoogleDriveClient(config.google_credentials_path)
        self.evaluator = ImageEvaluator(
            host=config.ollama_host,
            model=config.ollama_model,
            default_prompt=config.evaluation_prompt,
        )

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Using model: {self.config.ollama_model}")

    async def on_message(self, message: discord.Message):
        """Handle incoming messages.

        Args:
            message: Discord message object.
        """
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Check if bot is mentioned
        if self.user not in message.mentions:
            return

        # Find Google Drive URLs in the message
        urls = self.gdrive.find_gdrive_urls(message.content)
        if not urls:
            await message.reply(
                "Google DriveのURLが見つかりませんでした。\n"
                "例: `@Bot https://drive.google.com/file/d/xxxxx/view`"
            )
            return

        # Process the first URL found
        url = urls[0]
        logger.info(f"Processing URL: {url}")

        # Send typing indicator and defer response
        async with message.channel.typing():
            try:
                result = await self._process_image(url)
                await self._send_response(message, result)
            except ValueError as e:
                await message.reply(f"エラー: {e}")
            except Exception as e:
                logger.exception("Error processing image")
                await message.reply(f"画像の処理中にエラーが発生しました: {e}")

    async def _process_image(self, url: str) -> str:
        """Download and evaluate an image.

        Args:
            url: Google Drive URL.

        Returns:
            Evaluation result text.
        """
        # Download image (run in thread to avoid blocking)
        image_data = await asyncio.to_thread(
            self.gdrive.download_image_from_url, url
        )
        logger.info(f"Downloaded image: {len(image_data)} bytes")

        # Evaluate image
        result = await self.evaluator.evaluate(image_data)
        logger.info("Image evaluation completed")

        return result

    async def _send_response(self, message: discord.Message, result: str):
        """Send evaluation result, splitting if necessary.

        Args:
            message: Original Discord message.
            result: Evaluation result text.
        """
        # Discord message limit is 2000 characters
        max_length = 2000

        if len(result) <= max_length:
            await message.reply(result)
        else:
            # Split into multiple messages
            chunks = [result[i:i + max_length] for i in range(0, len(result), max_length)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)


def create_bot(config: Config) -> ImageEvaluationBot:
    """Create a configured bot instance.

    Args:
        config: Application configuration.

    Returns:
        Configured bot instance.
    """
    return ImageEvaluationBot(config)
