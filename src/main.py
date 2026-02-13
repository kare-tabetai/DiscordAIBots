"""Entry point for Discord AI Bot."""

import logging
import sys

from .bot import create_bot
from .config import Config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Run the Discord bot."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = Config.load()
        logger.info("Configuration loaded successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    bot = create_bot(config)

    logger.info("Starting Discord bot...")
    bot.run(config.discord_token)


if __name__ == "__main__":
    main()
