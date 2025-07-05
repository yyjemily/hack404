// 1. React core
import React from 'react';
import { createRoot } from 'react-dom/client';

// 2. React Router
import {
  createBrowserRouter,
  RouterProvider,
} from 'react-router-dom'

createRoot(document.getElementById('root')).render(
  <Providers>
    <RouterProvider router={router} />
  </Providers>
);

// 3. Global CSS
import './styles.css';

// 4. App pages (for your <RouterProvider>)
import Home from './pages/Home';



const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />
  }
]);
