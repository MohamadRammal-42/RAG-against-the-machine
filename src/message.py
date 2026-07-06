def prepare_message(question, filtered_text):
    system_instruction = """
    You are a technical documentation QA assistant.

    Use ONLY the provided context.

    Rules:
    - Output ONLY the final answer extract it from the context.
    - Never show your reasoning.
    - Never output <think>, reasoning traces, or analysis.
    - Keep the answer to one or two sentences.
    - If the answer is not explicitly present in the context, reply exactly:
    "The provided context does not contain the answer."
    """
    message = f"""Context:
    {filtered_text}

    Question:
    {question}
    """

    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": message,
        },
    ]
    return messages
