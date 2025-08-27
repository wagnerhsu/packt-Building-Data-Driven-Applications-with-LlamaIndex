import requests
import json
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

def test_lm_studio_connection():
    """Test LM Studio connection and diagnose issues"""
    base_url = "http://localhost:1234"

    print("Testing LM Studio connection...")

    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✓ LM Studio server is running")
            models = response.json()
            print(f"Available models: {len(models.get('data', []))}")
        else:
            print(f"✗ Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to LM Studio: {e}")
        print("Make sure LM Studio is running on http://localhost:1234")
        return False

    # Test 2: Test chat completions endpoint
    try:
        test_payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "temperature": 0.7
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            print("✓ Chat completions endpoint is working")
            return True
        else:
            print(f"✗ Chat completions failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Chat completions request failed: {e}")
        return False

def configure_optimized_llm():
    """Configure LlamaIndex with optimized settings for LM Studio"""
    Settings.llm = OpenAI(
        api_key="not-needed",
        base_url="http://localhost:1234/v1",
        timeout=120.0,  # Increased timeout
        max_retries=2,  # Reduced retries to fail faster
        # Additional parameters to reduce retry issues
        model="local-model",
        temperature=0.7,
        max_tokens=512
    )
    print("✓ Configured LlamaIndex with optimized settings")

if __name__ == "__main__":
    if test_lm_studio_connection():
        configure_optimized_llm()
        print("\nLM Studio is ready for use!")
    else:
        print("\nPlease check LM Studio setup:")
        print("1. Make sure LM Studio is running")
        print("2. Load a model in LM Studio")
        print("3. Start the local server (usually on port 1234)")
