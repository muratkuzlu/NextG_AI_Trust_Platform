import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import Register from './components/Register';
import Loaddata from './pages/Loaddata';
import PageNotFound from './pages/PageNotFound'
import ClassLogin from './components/ClassLogin'
import LoginPage from './components/LoginPage'
import reportWebVitals from './reportWebVitals';
import 'bootstrap/dist/css/bootstrap.min.css';
import {Route, BrowserRouter, Routes} from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './components/Login';

ReactDOM.render(
  <React.StrictMode>
    <BrowserRouter>

    <Navbar />
      <Routes>
        {/* <Route index element={<ClassLogin />} /> */}
        <Route index element={<Login />} />
        <Route path="app" element={<App />} />
        <Route path="/register" element={<Register />} />
        <Route path="/loaddata" element={<Loaddata />} />
        {/* <Route path="/home" element={<HomeView />} /> */}
        <Route path="/*" element={<PageNotFound />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
