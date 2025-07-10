#!/usr/bin/env python3
"""Test compression functionality"""

from src.context_switcher_mcp.compression import compress_perspectives, prepare_synthesis_input

# Test with large perspective responses
test_perspectives = {
    "technical": "A" * 3000,  # 3000 chars
    "business": "B" * 3000,
    "user": "C" * 3000,
    "risk": "D" * 3000,
    "philosophical": "E" * 3000
}

print("Test perspectives:")
for name, content in test_perspectives.items():
    print(f"  {name}: {len(content)} chars")

print("\nCompressing to 2000 chars each...")
compressed = compress_perspectives(test_perspectives, max_chars_per_perspective=2000)

print("\nCompressed sizes:")
for name, content in compressed.items():
    print(f"  {name}: {len(content)} chars")

print("\nPreparing synthesis input...")
synthesis_input = prepare_synthesis_input(test_perspectives, max_total_chars=12000)
print(f"Total synthesis input: {len(synthesis_input)} chars")
print(f"Estimated tokens: {len(synthesis_input) // 4}")
