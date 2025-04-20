import { useState } from 'react';
import './App.css';

enum Models {
  NAIVE_BAYES = 'naivebayes',
  SVM = 'svm',
  LOGISTIC_REGRESSION = 'lr',
}

const availableModels: Record<string, Models> = {
  'Naive Bayes': Models.NAIVE_BAYES,
  'Support Vector Machine': Models.SVM,
  'Logistic Regression': Models.LOGISTIC_REGRESSION,
};

type TopWord = {
  spam_score: number;
  word: string;
};

type PredictionResponse = {
  prediction: string;
  top_words: TopWord[];
};

const SERVER_BASE_URL = 'http://127.0.0.1:5000';

function App() {
  const [selectedModel, setSelectedModel] = useState<Models>(
    Models.NAIVE_BAYES
  );
  const [input, setInput] = useState('');

  const [prediction, setPrediction] = useState('');
  const [topWords, setTopWords] = useState<TopWord[]>([]);

  const sendInput = async () => {
    const res = await fetch(`${SERVER_BASE_URL}/${selectedModel}_predict`, {
      headers: {
        'Content-type': 'application/json',
        Accept: 'application/json',
      },
      method: 'POST',
      body: JSON.stringify({ message: input }),
    });
    const json = (await res.json()) as PredictionResponse;

    setPrediction(json.prediction);
    setTopWords(json.top_words);
  };

  return (
    <>
      <div>
        <h1>Spam Classifier</h1>
        <textarea
          placeholder="Enter spam message..."
          onChange={(e) => setInput(e.target.value)}
          value={input}
        />
        <div className="select-model-container">
          <h3>Select model</h3>
          {Object.keys(availableModels).map((modelName) => (
            <div key={modelName}>
              <input
                type="checkbox"
                name={modelName}
                checked={availableModels[modelName] === selectedModel}
                onChange={() => setSelectedModel(availableModels[modelName])}
              />
              <label htmlFor={modelName}>{modelName}</label>
            </div>
          ))}
        </div>
        <div>
          <button onClick={sendInput}>Submit</button>
        </div>
        {prediction.length > 0 ? (
          <>
            <hr />
            <h2>Result</h2>
            <h3 style={{ color: prediction === 'spam' ? 'red' : 'green' }}>
              {prediction}
            </h3>
            <div>
              <h3>Top Spam/Ham Words</h3>
              <ul>
                {topWords.map((entry) => (
                  <li key={entry.word}>
                    {entry.word}: {entry.spam_score}
                  </li>
                ))}
              </ul>
            </div>
          </>
        ) : (
          <></>
        )}
      </div>
    </>
  );
}

export default App;
