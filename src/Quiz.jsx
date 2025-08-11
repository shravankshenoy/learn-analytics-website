import { useState } from "react";

const quizData = [
  {
    question: "What is the purpose of the \"system\" message in the above snippet?",
    code: `messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"}
]`,
    options: [
      "It provides the assistant with background instructions.",
      "It contains the user's question.",
      "It is ignored by the model.",
      "It outputs the assistant's final answer."
    ],
    answer: 0
  },
  {
    question: "What does the third message (assistant role) represent?",
    code: `messages = [
  {"role": "system", "content": "You are a sarcastic assistant."},
  {"role": "user", "content": "How do I boil an egg?"},
  {"role": "assistant", "content": "Just throw it in the sun for a few hours."}
]`,
    options: [
      "A user's follow-up question",
      "A system instruction",
      "A model's previous response",
      "A hallucination"
    ],
    answer: 2
  },
  {
    question: "What kind of response will the assistant likely generate?",
    code: `messages = [
  {"role": "system", "content": "You are a formal and concise assistant."},
  {"role": "user", "content": "Tell me a joke."}
]`,
    options: [
      "Funny and lengthy story",
      "Very informal and humorous",
      "Sarcastic and mocking",
      "A short, polite joke"
    ],
    answer: 3
  },
  {
    question: "If we pass this to the LLM, what will it likely do next?",
    code: `messages = [
  {"role": "system", "content": "You are a tech support assistant."},
  {"role": "user", "content": "My printer won't work."},
  {"role": "assistant", "content": "Have you tried turning it off and on again?"},
  {"role": "user", "content": "Yes, still doesn't work."}
]`,
    options: [
      "Wait for user input",
      "Continue asking diagnostic questions",
      "Reboot the printer",
      "Summarize the conversation"
    ],
    answer: 1
  },
  {
    question: "If no \"system\" message is included, what will happen?",
    code: `messages = [
  {"role": "user", "content": "What's 2+2?"}
]`,
    options: [
      "The model crashes",
      "The model refuses to respond",
      "The model assumes a generic helpful assistant role",
      "The user has to define the assistant behavior manually"
    ],
    answer: 2
  },
  {
    question: "What mistake is being made in this prompt?",
    code: `messages = [
  {"role": "user", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What's the weather today?"}
]`,
    options: [
      "Two system messages are not allowed",
      "The assistant role is missing",
      "System instruction is given as a user message",
      "Nothing is wrong"
    ],
    answer: 2
  },
  {
    question: "Why include the \"assistant\" message manually?",
    code: `messages = [
  {"role": "user", "content": "Translate 'hello' to French."},
  {"role": "assistant", "content": "'Hello' in French is 'bonjour'."},
  {"role": "user", "content": "Now translate 'goodbye'."}
]`,
    options: [
      "To trick the model",
      "To simulate chat history and maintain context",
      "To confuse the LLM",
      "It is mandatory in every request"
    ],
    answer: 1
  }
];

/***

TypeError: crypto.hash is not a function
https://github.com/vitejs/vite/issues/20287
 I had a version 18 of node and when I upgraded to version 22.16.0 I deleted the node_module files, the package-lock.json and I reinstalled the dependencies, it now works without any problem


To upgrade node js
https://stackoverflow.com/questions/10075990/upgrading-node-js-to-the-latest-version
Downloaded msi file from https://nodejs.org/en/download/

Deplyoing app on Netlify
You build your app into build folder(npm run build) and literally just drag and drop build folder on netlify and thats it.


export declaration is used to make variables, functions, or classes available for use in other modules or files

There are two primary ways to use export:
- Named export : Used to export multiple values from a module.
- Default export : Used to export a single, primary value from a module.
- We can import a default export with any name, but named export must be imported with the same name

Below is an example of default export

Why do we need React in the first place? 
The real value for react (and other frameworks like it) comes from the reusable components and the single page aspects. While you can accomplish this pure JavaScript you will waste a lot of time building the structure and backbone of your application.

What is jsx?
Allows developers to write HTML-like code directly within their JavaScript files, thus allowing for the integration of markup and logic within the same JavaScript file. JSX code must be transpiled into regular JavaScript using tools like Babel before it can be executed in a web browser.

What are arrow functions?
Arrow functions were introduced in ES6. Arrow functions allow us to write shorter function syntax.
Normal function -> Anonymous function -> Arrow function

What is this binding?
 "this" binding refers to the process of determining the value of the this keyword within a function's execution context. The value of this is not fixed when a function is defined; instead, it is determined at the time the function is called and depends on how that call is made

Arrow functions handle this lexically. They do not have their own this binding; instead, they inherit this from their enclosing scope. This means this inside an arrow function will always be the same as this in the code that immediately surrounds it.

const obj = {
      value: 42,
      getValue: () => {
        console.log(this.value); // 'this' here refers to the global object (or undefined in strict mode)
      }
How does it help state management? What is prop drilling?

Prototype is an object that serves as a template from which other objects can inherit properties and methods.  Every object in JavaScript has an internal property called [[Prototype]] (or __proto__), which links it to another object, its prototype
Classes (introduced in ES6) are primarily syntactic sugar over the existing prototype-based inheritance model
"Prototypal inheritance" is an unofficial term that the community invented to describe the delegation behavior (when you try to access a property or method on an object, JavaScript first checks if that object itself has the property. If not, it then looks up the prototype chain, following the [[Prototype]] link, until it finds the property or reaches the end of the chain which is usually null) that we believed at the time to be unique to JavaScript. But actually it turns out delegation isn't unique to JavaScript at all. Nor is it unique to prototypes, since class inheritance often uses delegation.
https://www.reddit.com/r/learnjavascript/comments/16y8b2b/inheritance_vs_classical_is_this_a_succinct/


https://www.reddit.com/r/webdev/comments/8fvasq/what_is_the_benefit_of_react_vs_pure_javascript/
https://www.reddit.com/r/reactjs/comments/17om8v9/what_is_the_point_of_state_management/
***/

export default function LLMRoleQuiz() {
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [score, setScore] = useState(0);

  const handleOptionClick = (i) => {
    if (showAnswer) return; // if user clicks second time after answer has been shown, then just return
    setSelected(i);
    setShowAnswer(true);
    if (i === quizData[current].answer) {
      setScore(score + 1);
    }
  };
  

  const nextQuestion = () => {
    setCurrent(current + 1);
    setSelected(null);
    setShowAnswer(false);
  };

  if (current >= quizData.length) {
    return (
      <div style={{ padding: "1.5rem", textAlign: "center" }}>
        <h2 style={{ fontSize: "1.5rem", fontWeight: "bold", marginBottom: "1rem" }}>Quiz Complete!</h2>
        <p style={{ fontSize: "1.2rem" }}>Your score: {score} / {quizData.length}</p>
      </div>
    );
  }

  const q = quizData[current];

  return (
    <div style={{ padding: "1.5rem", maxWidth: "800px", margin: "0 auto" }}>
      <div style={{ border: "1px solid #ccc", borderRadius: "8px", marginBottom: "1rem" }}>
        <div style={{ padding: "1rem" }}>
          <pre style={{ backgroundColor: "#f0f0f0", padding: "0.75rem", borderRadius: "4px", overflowX: "auto" }}>
            {q.code}
          </pre>
          <h2 style={{ fontSize: "1.25rem", fontWeight: "bold", marginTop: "1rem", marginBottom: "0.5rem" }}>Q{current + 1}: {q.question}</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {q.options.map((opt, i) => {
              let bgColor = "#fff";
              if (showAnswer) {
                if (i === q.answer) bgColor = "#d4edda";
                else if (i === selected) bgColor = "#f8d7da"; // if the correct answer is not selected then use this background color
              }
              return (
                <button
                  key={i}
                  onClick={() => handleOptionClick(i)}
                  style={{
                    backgroundColor: bgColor,
                    border: "1px solid #ccc",
                    padding: "0.5rem 1rem",
                    borderRadius: "4px",
                    textAlign: "left",
                    cursor: showAnswer ? "default" : "pointer"
                  }}
                >
                  {opt}
                </button>
              );
            })}
          </div>
        </div>
      </div>
      {showAnswer // show next button only if showAnswer is true
      && (
        <button
          onClick={nextQuestion}
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Next
        </button>
      )}
    </div>
  );
}
