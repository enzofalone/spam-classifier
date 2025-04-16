import { useState } from 'react';
import './App.css';

enum Models {
  NAIVEBAYES = 'naivebayes',
}

const availableModels: Record<string, Models> = {
  'Naive Bayes': Models.NAIVEBAYES,
};

function App() {
  const [selectedModel, setSelectedModel] = useState<Models>(Models.NAIVEBAYES);
  const [input, setInput] = useState('');

  const sendInput = () => {};

  return (
    <>
      <div>
        <h1>Spam Classifier</h1>
        <textarea onChange={(e) => setInput(e.target.value)} value={input} />
        <div className="select-model-container">
          <h3>Select model</h3>
          {Object.keys(availableModels).map((modelName) => (
            <>
              <input
                type="checkbox"
                name={modelName}
                checked={availableModels[modelName] === selectedModel}
                onClick={() => setSelectedModel(availableModels[modelName])}
              />
              <label htmlFor={modelName}>{modelName}</label>
            </>
          ))}
        </div>
        <div>
          <button onClick={sendInput}>Submit</button>
        </div>
      </div>
    </>
  );
}

export default App;
