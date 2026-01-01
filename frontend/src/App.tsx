import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="App">
      <h1>AI Clone Application</h1>
      <p>Welcome to your AI Clone app!</p>
      <p>Backend API: <a href="http://localhost:8000/docs" target="_blank">http://localhost:8000/docs</a></p>
    </div>
  )
}

export default App

