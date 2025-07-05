// 1. React core
import React, { StrictMode } from "react";
import { createRoot } from 'react-dom/client';

// 2. React Router
import {
  createBrowserRouter,
  RouterProvider,
} from 'react-router-dom'


// 3. Global CSS
import './styles.css';

// 4. App pages (for your <RouterProvider>)
import Home     from './pages/Home.jsx';
import Chatbot  from './pages/Chatbot.jsx';


const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />
  },
  {
    path: "/chatbot",
    element: <Chatbot />
  }
]);
