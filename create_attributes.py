from openai import OpenAI
import yaml

client = OpenAI()
# MODEL = 'gpt-3.5-turbo'
MODEL = 'gpt-4'

prompt = """
# You are a/an {} expert. What attributes can distinguish images by one of the following <Class Lists>? Answer according to <Answer Format> below.

<Answer Format>
- <attribute_1>: <example_value_1_1>, <example_value_1_2>
- <attribute_2>: <example_value_2_1>, <example_value_2_2>, <example_value_2_3>
- ...

<Class Lists>
- {}
""".strip()

field = 'aircraft'
classes = yaml.safe_load(open('data/CoOp/FGVCAircraft/classes.yaml'))
prompt = prompt.format('aircrafts', ', '.join(classes))

print(prompt)
print()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model=MODEL,
    temperature=0
)

print(chat_completion.choices[0].message.content)
