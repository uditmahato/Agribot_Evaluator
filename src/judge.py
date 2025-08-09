from openai import OpenAI

def make_client(api_key: str):
    return OpenAI(api_key=api_key)

def judge_pairwise(client, model: str, system_prompt: str, lang: str, question: str, A_text: str, B_text: str):
    user_prompt = (
        f"Question (language = {lang}): {question}\n"
        f"Answer A: {A_text}\n"
        f"Answer B: {B_text}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
