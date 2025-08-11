## LLM Roles
### Basics
When you interact with ChatGpt, you ask a question, and ChatGpt replies. The question you ask is called **user prompt** and the response given by ChatGpt is called **assistant prompt**. 
\
Besides the above roles (user and assistant), there is also a third role called system role which is useful as a developer. A role basically helps define interactions. A **system prompt** defines how the LLM behaves.
\
Below is an example of the 3 kinds of roles:
```
messages = [
  {"role": "system", "content": "You are a sarcastic assistant."},
  {"role": "user", "content": "How do I boil an egg?"},
  {"role": "assistant", "content": "Just throw it in the sun for a few hours."}
]

```

### Python implementation
You might be wondering how this is actually implemented in Python. It would be as follows: 
1. When user inputs a query or response, it is stored into conversation history with role user
2. LLM response is stored with role assistant

Below is a code example:

```
from groq import Groq

client = Groq(
    api_key=GROQ_API_KEY,
)

# variable to store conversation history
messages = [
     {"role": "system", "content": "You are a sarcastic assistant who gives short but sarcastic answers"}
]

while True:
  user_input = input("Enter your query or response ")
  
  if user_input == "quit":
    break
  else:
    messages.append({"role":"user", "content":user_input})
    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile"
    )

    messages.append({"role":"assistant", "content":response.choices[0].message.content})
    print(response.choices[0].message.content)

print(messages)
```

Try implementing the above code in the Jupyter notebook below : https://colab.research.google.com/drive/1iiAyo2IOE4YBqlmEmP3GVk8fb8R77MS3?usp=sharing