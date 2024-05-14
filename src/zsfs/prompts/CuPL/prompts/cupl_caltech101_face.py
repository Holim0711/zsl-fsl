import openai

client = openai.OpenAI()

# MODEL = 'gpt-3.5-turbo-0125'  # text completion not working
# MODEL = 'gpt-3.5-turbo-0613'  # text completion not working
# MODEL = 'gpt-3.5-turbo-1106'  # text completion not working
# MODEL = 'gpt-3.5-turbo-0301'  # text completion not working
MODEL = 'gpt-3.5-turbo-instruct-0914'


def cupl_code(prompt: str):
    response = openai.Completion.create(    # deprecated
        engine="text-davinci-002",
        prompt=prompt,
        temperature=.99,
        max_tokens=50,
        n=10,
        stop="."
    )
    results = []
    for r in range(len(response["choices"])):
        result = response["choices"][r]["text"]
        results.append(result.replace("\n\n", "") + ".")
    return results


def get_text_completion(prompt: str):
    text_completion = client.completions.create(
        model=MODEL,
        prompt=prompt,
        temperature=0.99,
        max_tokens=50,
        stop='.',
        n=10
    )
    return [x.text.strip() for x in text_completion.choices]


prompts = [
    'Describe what a face looks like:',
    'Describe a face:',
    # 'What are the identifying characteristics of a face?',
    'What are the identifying characteristics of a face in a short sentence?',
]

for x in prompts:
    for y in get_text_completion(x):
        print(y + '.')
