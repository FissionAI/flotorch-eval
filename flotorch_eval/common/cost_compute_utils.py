import os
from typing import Dict, Any
from decimal import Decimal
from dataclasses import dataclass
import json
import time
import httpx
from pathlib import Path

MILLION = 1_000_000
THOUSAND = 1_000
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
MODEL_PRICING_API_URL = "https://models.dev/api.json"

@dataclass
class MetricsData:
    """
    Data class to store metrics information.

    Attributes:
        cost (Decimal): The cost associated with the operation.
        latency (float): The latency in milliseconds.
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
    """
    cost: Decimal
    latency: float
    input_tokens: int
    output_tokens: int

def get_cache_directory():
    """
    Get the path to the cache directory for storing the price list JSON file.

    Returns:
        Path: Path object pointing to the price_list.json file.
    """
    try:
        DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        DATA_DIR = Path.cwd() / "flotorch_eval" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "price_list.json"

class PriceCache:
    """
    Handles fetching and caching of model price lists from an external API.

    This class can get the price list from the API and cache it in a file.
    It can also get the model cost from the price list.

    Attributes:
        cache_file (Path): Path to the JSON file for persistent caching.
        ttl (int): Time-to-live for cache in seconds.
        _cache (Dict[str, Any]): In-memory cache of the price list.
        _last_fetched (float): Timestamp of the last fetch.
    """
    def __init__(self, cache_file=get_cache_directory(), ttl=3600):
        """
        Initialize the PriceCache.

        Args:
            cache_file (Path, optional): Path to JSON file for persistent caching.
            ttl (int, optional): Time-to-live for cache in seconds (default: 1 hour).
        """
        self.cache_file = cache_file
        self.ttl = ttl
        self._cache: Dict[str, Any] = {}
        self._last_fetched = 0

    def _load_from_file(self) -> bool:
        """
        Load the price list from the cache file if it exists.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
                return True
            except Exception:
                return False
        return False

    def _save_to_file(self):
        """
        Save the current in-memory cache to the cache file.
        """
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    async def _fetch_from_api(self) -> Dict[str, Any]:
        """
        Fetch the price list from the external API.

        Returns:
            Dict[str, Any]: Raw price list data from the API.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = MODEL_PRICING_API_URL
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    def _process_price_list(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten provider-based list into a model-based list for quicker lookup.

        Args:
            raw (Dict[str, Any]): Raw price list data from the API.

        Returns:
            Dict[str, Any]: Flattened price list keyed by model_id.
        """
        flat = {}
        for provider_name, pdata in raw.items():
            for model_id, mdata in (pdata.get("models") or {}).items():
                flat[model_id] = {
                    "provider": provider_name,
                    **mdata,
                }
        return flat

    async def get_price_list(self, force_refresh=False) -> Dict[str, Any]:
        """
        Get the price list, using cache if available and valid.

        Args:
            force_refresh (bool, optional): If True, force refresh from API.

        Returns:
            Dict[str, Any]: Flattened price list keyed by model_id.
        """
        # Use in-memory cache if valid
        if (
            not force_refresh
            and self._cache
            and (time.time() - self._last_fetched < self.ttl)
        ):
            return self._cache

        # Load from disk if available
        if not force_refresh and self._load_from_file():
            return self._cache

        # Fetch from API and process
        raw = await self._fetch_from_api()
        self._cache = self._process_price_list(raw)
        self._last_fetched = time.time()
        self._save_to_file()
        return self._cache

    async def get_model_cost(self, model_id: str, force_refresh=False) -> Dict[str, Any]:
        """
        Get the cost information for a specific model.

        Args:
            model_id (str): The model identifier.
            force_refresh (bool, optional): If True, force refresh from API.

        Returns:
            Dict[str, Any]: Cost information for the specified model.

        Raises:
            ValueError: If the model_id is not found in the price list.
        """
        price_list = await self.get_price_list(force_refresh=force_refresh)
        if model_id not in price_list:
            raise ValueError(f"Model '{model_id}' not found in pricing data")
        return price_list[model_id]["cost"]


def extract_metadata_metrics(metadata: Dict[str, Any]) -> MetricsData:
    """
    Extract metrics from a metadata dictionary.

    Args:
        metadata (Dict[str, Any]): Dictionary containing metadata information.

    Returns:
        MetricsData: Object with extracted metrics (input_tokens, output_tokens, latency, cost).
    """
    return MetricsData(
        input_tokens=metadata.get("inputTokens", 0),
        output_tokens=metadata.get("outputTokens", 0),
        latency=float(metadata.get("latencyMs", 0)),
        cost=Decimal('0.0000')
    )

async def calculate_model_inference_cost(input_tokens, output_tokens, inference_model):
    """
    Calculate the cost of model inference based on input and output tokens.

    Args:
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
        inference_model (str): Model identifier.

    Returns:
        float: Total cost of inference for the given tokens and model.
    """
    price_cache = PriceCache()
    cost = await price_cache.get_model_cost(inference_model)
    input_price = float(cost['input']) * input_tokens / MILLION
    output_price = float(cost['output']) * output_tokens / MILLION
    return input_price + output_price
