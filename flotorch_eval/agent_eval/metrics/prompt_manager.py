import os
from typing import Dict
from pathlib import Path

class PromptManager:
    """
    Manages loading and retrieving prompt templates from a directory.
    """
    def __init__(self):
        """
        Initializes the PromptManager.

        Args:
            prompt_dir (str): The directory where prompt files are stored.
        """
        current_dir = Path(__file__).parent
        self.prompt_dir = current_dir / "prompts"
        self._cache: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self):
        """
        Loads all .txt files from the prompt directory into the cache.
        The filename (without extension) is used as the prompt name.
        """
        if not os.path.isdir(self.prompt_dir):
            print(f"Warning: Prompt directory '{self.prompt_dir}' not found.")
            return

        for filename in os.listdir(self.prompt_dir):
            if filename.endswith(".txt"):
                prompt_name = os.path.splitext(filename)[0]
                filepath = os.path.join(self.prompt_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self._cache[prompt_name] = f.read()
                except Exception as e:
                    print(f"Error loading prompt {prompt_name}: {e}")

    def get_prompt(self, name: str) -> str:
        """
        Retrieves a prompt template by its name.

        Args:
            name (str): The name of the prompt (corresponds to the filename).

        Returns:
            str: The prompt template string.

        Raises:
            ValueError: If the prompt with the given name is not found.
        """
        prompt = self._cache.get(name)
        if not prompt:
            raise ValueError(f"Prompt '{name}' not found. Ensure a '{name}.txt' file exists in the '{self.prompt_dir}' directory.")
        return prompt
