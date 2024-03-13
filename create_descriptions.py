from openai import OpenAI
import yaml

client = OpenAI()
# MODEL = 'gpt-3.5-turbo'
MODEL = 'gpt-4'

PROMPT = """
# You are a/an {} expert. Write single-sentence descriptions of '{}' according to the <Attribute Lists> below. Answer according to <Answer Format> below.

<Answer Format>
- <attribute_1>: <single-sentence description>
- <attribute_2>: <single-sentence description>
- ...

<Attribute Lists>
{}
""".strip()

field = 'aircraft'
classes = yaml.safe_load(open('data/CoOp/FGVCAircraft/classes.yaml'))

attributes = [x[1:].strip() for x in open('data/Desc/FGVCAircraft/attributes.txt')]
attributes = [x.split(':') for x in attributes]
attributes = [(x, y.split(', ')) for x, y in attributes]
attributes = [f'- {x}: ' + ', '.join(y[:3]) + ', ...'  for x, y in attributes]
attributes = '\n'.join(attributes)

results = []

for i, c in enumerate(classes):
    if i < 72:
        continue
    prompt = PROMPT.format('aircrafts', c, attributes)

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

    text = chat_completion.choices[0].message.content
    print(text)
    with open(f'data/Desc/FGVCAircraft/description/{i:03d}.{c.replace(' ', '_').replace('/', '_')}.txt', 'w') as file:
        print(text, file=file)
