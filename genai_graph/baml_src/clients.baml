// Default BAML LLM clients for document outline extraction

client<llm> CustomClient {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "deepseek/deepseek-v4-flash"
    temperature 0.0
  }
}
