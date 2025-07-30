# Prompt Engineering Overview

### Introduction
**Prompt** is instruction we give to language model

**Prompt Engineering** is the practice of designing effective instructions(i.e. prompts) for language models like ChatGPT, Gemini to get the desired output. It involves choosing the right instructions, format, and examples to guide the model's behavior.


### Zero-Shot Prompting

- **Definition**: The model is given a task without any examples.
- **Use Case**: When the task is simple or well-known to the model.
- **Example**: Translate the following sentence to French: "How are you?"

### Few-shot prompting
**Definition**: The model is given a few examples to understand the task pattern.
- **Use Case**: When the task is complex or less common.
- **Example**: Translate the following sentences to French:\
"Good morning" → "Bonjour"\
"Thank you" → "Merci"\
"I am tired" →

A more real world example:
```
Classify customer emails into one of these categories: Billing, Technical Issue, Feedback.

### Examples
Email: "I was charged twice for my last purchase. Please look into this."
Category: Billing

Email: "The app crashes every time I try to upload a photo."
Category: Technical Issue

Email: "I really love the new update! Great job."
Category: Feedback

Classify the following email : "I can't reset my password. The link doesn't work."

```