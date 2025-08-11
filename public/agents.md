## What are agents?
An agent is a LLM with state and tools. Lets understand this better. An AI agent is made of 3 things:
1. LLM : Models like GPT, Llama, Claude which act as brain of agent
2. Memory/State : An LLM is by default does not have memory i.e. it cannot remember previous conversations. An agent adds memory by using some datastore to store the history (i.e. previous conversations)
3. Tools : Tool is a function/add on that enables agent to perform an action (beyond its inherent limitations). Some examples of tools can be a querying a database, web search, an API call, etc

If we compare agent to a human being, LLM and memory are like the brain which do the thinking, and tools are like the hands and legs which performs what the brain says

## Workflow vs Agent
* Workflow : Workflows are predefined sequences of steps performed one after another
* Agent : Systems that can make their own decisions and dynamically adjust their actions based on the situation and available tools

Example : Lets say we create a travel booking agent. A workflow would be the following steps in sequence
1. Check flight dates
2. Check weather on dates
3. Check hotel availability dates
4. Check car rentals
5. Decide places to visit on each day

However in reality, based on hotel availability dates, we might want to recheck and changes the flight dates. Or based on order of places to visit, we might want to check a new hotel dates. So it is very complicated to capture this entire thing in a workflow. So have 1 LLM and all of the other things as tools (flight tool, weather tool, hotel tool, excursion tool, etc). LLM decides which tool to use and call the tools in any order and multiple times based on user requirements.





### References
1. https://www.youtube.com/watch?v=kmwWSBJGewM&t=71s (What is an AI Agent)
2. https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/