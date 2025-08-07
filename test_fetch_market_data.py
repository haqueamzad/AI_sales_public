import asyncio
import logging
from trading_env import fetch_market_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Testing fetch_market_data for ('NVDA',)")
    df = await fetch_market_data(('NVDA',), '2025-01-01', '2025-01-31')
    print(df.to_string(index=False))

if __name__ == '__main__':
    asyncio.run(main())