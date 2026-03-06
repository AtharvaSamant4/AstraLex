"""Test both API keys independently to confirm they work."""
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

keys = {
    "Key 1 (GEMINI_API_KEY)": os.getenv("GEMINI_API_KEY", ""),
    "Key 2 (GEMINI_API_KEY_2)": os.getenv("GEMINI_API_KEY_2", ""),
}

for label, key in keys.items():
    if not key:
        print(f"{label}: NOT SET")
        continue
    print(f"\n{label}: {key[:10]}...{key[-4:]}")
    client = genai.Client(api_key=key)
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello in one word.",
            config=genai.types.GenerateContentConfig(
                max_output_tokens=10,
            ),
        )
        print(f"  ✓ WORKS — response: {(resp.text or '').strip()[:50]}")
    except Exception as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg:
            print(f"  ⚠ QUOTA EXHAUSTED (key is valid but daily limit hit)")
        else:
            print(f"  ✗ ERROR: {msg[:120]}")

# Also verify pipeline picks up both
from rag.model_manager import ModelManager
ModelManager.reset()
from rag.pipeline import RAGPipeline
pipe = RAGPipeline()
print(f"\nPipeline registered {ModelManager.total_keys()} key(s)")
print(f"Active key index: {ModelManager.active_key_index() + 1}")
