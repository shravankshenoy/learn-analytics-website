import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import nlpbasics from "./../public/nlp-basics.md";

/*
"Side Effect" is not a react-specific term. It is a general concept about behaviours of functions. A function is said to have side effect if it trys to modify anything outside its body. For example, if it modidifies a global variable, then it is a side effect. If it makes a network call, it is a side effect as well.

https://www.reddit.com/r/reactjs/comments/8avfej/what_does_side_effects_mean_in_react/

Component rendering in React refers to the process by which React takes your component code and translates it into what is displayed on the user's screen. When your application first loads, React starts from the root component and works its way down the component tree. JSX within your components is converted into React elements using React.createElement. These elements are JavaScript objects representing the desired UI structure. Subsequent renders are triggered when a component's state or props change. In commit phase, React interacts with the browser's DOM. Only the necessary parts of the DOM are manipulated,


Which is the root component in React?
root component is the top-level component that serves as the entry point for your entire application's component hierarchy. Using ReactDOM.createRoot().render() (in React 18+), the root component is "mounted" onto a specific HTML element in your index.html file : <script type="module" src="/src/main.jsx"></script>


Side effects refer to any operations or behaviors that occur in a component after rendering, and that don’t directly impact the current component render cycle. Can include tasks such as data fetching, network calls, fetching data from an API asynchronously

When you begin writing code in React to fetch data from an API, you may encounter an issue where your application enters into an infinite rendering loop. This occurs because updating the state triggers a re-render, leading to the execution of the same code again. By using the useEffect hook we can make sure that API call happens only when a value changes rather than every single time. In the file loading case below, we load the file only when the file changes. The second argument in the useEffect hook is the dependency array, i.e. when the function should run

https://gurindernarang.medium.com/side-effect-useeffect-in-react-2dc6cdf0b9c5

How does fetch read a file in js? Also why do we need to use aync-await while fetching file?


What are props?
"props" (short for properties) are a way of passing data from one component to another, typically from a parent component to its child components.  They are essentially arguments that you pass to a component. Here we pass the file name as a prop

Error:  require is not defined

Error: 'import ... =' can only be used in TypeScript files import nlp-basics from "./nlp-basics.md";
Problem : import nlp-basics from "./../public/nlp-basics.md";
Solution : import nlpbasics from "./../public/nlp-basics.md";

Failed to parse source for import analysis because the content contains invalid JS syntax. You may need to install appropriate plugins to handle the .md file format, or if it's an asset, add "** /*.md" to `assetsInclude` in your configuration.
https://stackoverflow.com/questions/73459654/import-markdown-files-dynamically-with-vite

Assertion: Unexpected value `[object Promise]` for `children` prop, expected `string`

Failed to execute 'text' on 'Response': Illegal invocation
Problem : const text = await response.text;
Solution : const text = await response.text();

*/


function MarkdownViewer(props){

    const [content, setContent] = useState("");

    useEffect(function() {
        async function loadMarkdown(){
            try{
                const response = await fetch(props.file);
                const text = await response.text();
                setContent(text);
            } catch(error){
                setContent("Failed to load markdown file.");
            }
        }

        if(true){
            loadMarkdown();
        }
    }, [])

    return (
        <div style={{ padding: "1rem" }}>
            <ReactMarkdown>{content}</ReactMarkdown>
        </div>
    );

}

export default MarkdownViewer;

