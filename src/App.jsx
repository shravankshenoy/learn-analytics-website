import React from "react";
import Quiz from "./Quiz";
import {Link} from "react-router-dom"
import MarkdownViewer from "./page";

/*
link is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`.
The <link> tag is a void element used to define the relationship between the current document and an external resource. Common uses include linking to stylesheets (<link rel="stylesheet" href="styles.css">) or favicons. As a void element, it does not have an explicit closing tag and should not contain any content within it.
Since void elements, by definition, cannot contain content, attempting to place children within a <link> tag (e.g., <link>Some text</link>) will result in an error or warning because it violates the HTML specification for void elements.
Reason : Forgot to import Link from react-router-dom
Solution : import {Link} from "react-router-dom"

react router dom does not provide an export named 'default'
Problem : import Link from "react-router-dom"
Solution : import {Link} from "react-router-dom"
*/ 

function App() {
  return (
    <div>
      <h1 style={{ textAlign: "center", marginTop: "1rem" }}>Main Page</h1>
      <Link to={"/nlp-basics"}>NLP Basics</Link>
      <Link to={"/gen-ai-quiz"}><div>Gen AI Quiz</div></Link>
      <Link to={"/prompt-engineering"}><div>Prompt Engineering</div></Link>
      
    </div>
  );
}

export default App;