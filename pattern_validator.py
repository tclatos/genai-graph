import re

# Regex pattern to match the specified format:
# - 1 or 2 slashes
# - Each part more than 5 characters (except last part which is more than 10 characters)
# - Parts contain alphanumeric characters, underscores, hyphens, or periods
PATTERN = r'^([a-zA-Z0-9_.-]{6,}/){1,2}[a-zA-Z0-9_.-]{11,}$'

def validate_pattern(text: str) -> bool:
    """
    Validate if a text matches the pattern with 1-2 slashes and length requirements.
    
    Args:
        text: String to validate
        
    Returns:
        True if the text matches the pattern, False otherwise
    """
    return bool(re.match(PATTERN, text))

# Test the function with provided examples
if __name__ == "__main__":
    # Valid examples
    test_cases = [
        "azure_ai/mistral-document-ai-2505",
        "openrouter/google/palm-2-chat-bison",
        "openrouter/openai/gpt-4.1-mini"
    ]
    
    print("Testing valid examples:")
    for case in test_cases:
        result = validate_pattern(case)
        print(f"'{case}' -> {result}")
    
    # Edge cases
    print("\nTesting edge cases:")
    edge_cases = [
        "short/mistral-document-ai-2505",  # First part too short
        "azure_ai/short-name",             # Last part too short
        "azure_ai/mistral-document-ai-2505/extra",  # Too many slashes
        "single-part-only",                # No slashes
        "a/b/c/d",                         # Too many parts
    ]
    
    for case in edge_cases:
        result = validate_pattern(case)
        print(f"'{case}' -> {result}")