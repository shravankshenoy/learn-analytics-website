import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {createBrowserRouter, RouterProvider} from "react-router-dom"
import App from './App.jsx'
import Quiz from './Quiz.jsx'
import MarkdownViewer from "./page";
import nlpbasics from "/nlp-basics.md";

//import nlpbasics from "./../public/nlp-basics.md";
//console.log(nlpbasics)

/*
Uncaught Error: You must provide a non-empty routes array to createRouter
Solutipn : Put dictionary in an array

https://www.youtube.com/watch?v=c02YoWR9gSY

What does import nlpbasics from "./../public/nlp-basics.md"; actually mean?
Files in the public directory are served at the root path.
Instead of /public/nlp-basics.md, use /nlp-basics.md. (x2)

In Vite, you might place static assets in a designated "public" directory. This provides a url to the asset.
Since we have placed nlp-basics.md in public folder, we can directly access the file at http://localhost:5173/nlp-basics.md

How does vite create url for static asset?
Refer https://vite.dev/guide/assets

You will not "see" Webpack in a Vite project because Vite replaces Webpack.

What is import.meta.url?

3 ways to handle static assets in Vite
1. Importing Assets as URLs
Vite will process the imported asset and return its resolved public URL. That is why import nlpbasics from "./../public/nlp-basics.md"; works. Refer https://www.youtube.com/watch?v=KOmVGNWJuVA
2. Importing Assets with Query Parameters such as ?raw
3. Placing Assets in the public Directory

*/

const router = createBrowserRouter([
  {path:"/", element:<App/>},
  {path:"/gen-ai-quiz", element:<Quiz/>},
  {path:"/nlp-basics", element:<MarkdownViewer file={nlpbasics}/>},
  {path:"/prompt-engineering", element:<MarkdownViewer file="/prompt-engineering.md"/>}
]
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
