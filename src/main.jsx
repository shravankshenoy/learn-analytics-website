import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import Quiz from './Quiz.jsx'
import {createBrowserRouter, RouterProvider} from "react-router-dom"

/*
Uncaught Error: You must provide a non-empty routes array to createRouter
 Solutipn : Put dictionary in an array
 */

const router = createBrowserRouter([
  {path:"/", element:<App/>},
  {path:"/gen-ai-quiz", element:<Quiz/>}
]
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
