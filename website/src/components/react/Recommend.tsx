import React, { useState } from 'react';
import { sendRequest } from '~/utils/serverless';
import { Loader2 } from '../ui/icons/Loader2';

export default function Recommend({ ...props }) {
  const [value, setValue] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [predictions, setPredictions]: any = useState(null);

  const handleChange = (e) => {
    const val = e.target.value;
    if (parseInt(val) <= 322896) {
      setValue(val);
    }
  };

  const handleClick = async () => {
    console.log('Hello world', value);
    if (parseInt(value) >= 0) {
      setIsLoading(true);
      setPredictions(null);

      const res = await sendRequest(parseInt(value), setError, setIsLoading);
      if (res.predictions) {
        setPredictions(res.predictions);
      }
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="flex flex-row pb-5">
        <input
          type="number"
          min={1}
          max={322896}
          value={value}
          onChange={handleChange}
          onKeyUp={(e) => {
            if (e.code === 'Enter') {
              handleClick();
            }
          }}
          placeholder="Enter a number"
          className="rounded-l-lg p-3 min-w-[200px] border border-gray-300 h-12 "
        />

        <button onClick={handleClick} disabled={isLoading} className="btn-primary rounded-none h-12 rounded-r-lg">
          {/* Recommend */}
          {isLoading ? (
            <div className={`${!isLoading ? 'hide' : ''}`}>
              <Loader2 />
            </div>
          ) : (
            'Recommend'
          )}
        </button>
      </div>
      {error && (
        <>
          <br />
          <span className="text-red-500">{error}</span>
        </>
      )}

      {predictions && (
        <div className="mt-4 w-full max-w-4xl overflow-x-auto flex gap-4">
          <div className="bg-gray-100 p-4 rounded-lg shadow-md w-1/2">
            <h2 className="text-lg font-bold mb-2">Clicked Articles</h2>
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  <th className="border px-4 py-2">Article ID</th>
                  <th className="border px-4 py-2">Category ID</th>
                  <th className="border px-4 py-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {predictions.clicked_articles.map((article, index) => (
                  <tr key={index}>
                    <td className="border px-4 py-2">{article.article_id}</td>
                    <td className="border px-4 py-2">{article.category_id}</td>
                    <td className="border px-4 py-2">{article.score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="bg-gray-100 p-4 rounded-lg shadow-md w-1/2">
            <h2 className="text-lg font-bold mb-2">Top Recommendations</h2>
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  <th className="border px-4 py-2">Article ID</th>
                  <th className="border px-4 py-2">Category ID</th>
                  <th className="border px-4 py-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {predictions.top_recommendation.map((recommendation, index) => (
                  <tr key={index}>
                    <td className="border px-4 py-2">{recommendation.article_id}</td>
                    <td className="border px-4 py-2">{recommendation.category_id}</td>
                    <td className="border px-4 py-2">{recommendation.score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
