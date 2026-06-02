#!/usr/bin/env python3
"""
GPT-5.5 on Amazon Bedrock — Demo Script
========================================
Endpoint: https://bedrock-mantle.us-east-2.api.aws/openai/v1
Model:    openai.gpt-5.5
API:      OpenAI Responses API (client.responses.create)

Usage:
  export AWS_BEARER_TOKEN_BEDROCK="your-bedrock-api-key"
  python gpt55_bedrock_demo.py
"""

import os
import sys
import time

def main():
    # --- Authentication ---
    api_key = (
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("BEDROCK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not api_key:
        print("❌ ERROR: No API key found.")
        print("Please set one of:")
        print("  export AWS_BEARER_TOKEN_BEDROCK='your-key'")
        print("  export BEDROCK_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    # --- Import OpenAI SDK ---
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ ERROR: openai SDK not installed.")
        print("Run: pip install -U openai")
        sys.exit(1)

    # --- Configuration ---
    BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
    MODEL = "openai.gpt-5.5"

    # --- Initialize client ---
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    print("=" * 60)
    print("🚀 GPT-5.5 on Amazon Bedrock — Demo")
    print("=" * 60)
    print(f"  Endpoint:  {BASE_URL}")
    print(f"  Model:     {MODEL}")
    print(f"  Reasoning: effort = medium")
    print(f"  Text:      verbosity = low")
    print("=" * 60)

    # --- API Call ---
    prompt = "What are the top 3 benefits of using Amazon Bedrock for model inference? Answer in bullet points."
    print(f"\n📝 Prompt: {prompt}\n")
    print("⏳ Calling GPT-5.5...")

    start = time.time()
    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "developer",
                    "content": "You are a senior AWS solutions architect. Be concise and practical.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            reasoning={"effort": "medium"},
            text={"verbosity": "low"},
        )
        elapsed = time.time() - start

        # --- Output ---
        print(f"\n✅ Success! Response in {elapsed:.2f}s")
        print("-" * 60)
        print(f"  Response ID:   {response.id}")
        print(f"  Model:         {response.model}")
        if hasattr(response, "usage") and response.usage:
            print(f"  Input tokens:  {response.usage.input_tokens}")
            print(f"  Output tokens: {response.usage.output_tokens}")
        print("-" * 60)
        print("\n📤 Output:\n")
        print(response.output_text)
        print("\n" + "-" * 60)
        print("✅ STATUS: PASS")

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Failed after {elapsed:.2f}s")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error:      {e}")
        print("\n💡 Troubleshooting:")
        if "401" in str(e) or "403" in str(e):
            print("  → Check your API key is valid and has GPT-5.5 access in us-east-2")
        elif "404" in str(e):
            print("  → Check endpoint URL and model ID")
        elif "429" in str(e):
            print("  → Rate limited, retry after a moment")
        print("\n❌ STATUS: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
