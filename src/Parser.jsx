import { useState } from 'react';
import './App.css'; // Предположим, что стили находятся в этом файле

function ParserPage() {
  const [urlInput, setUrlInput] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Функция для обработки ввода URL
  function handleUrlChange(event) {
    setUrlInput(event.target.value);
  }

  // Функция для отправки запроса на сервер
  function handleSubmit() {
    if (!urlInput) {
      alert('Пожалуйста, введите ссылку.');
      return;
    }

    setLoading(true);

    const endpoint = 'http://176.124.215.11:8000/api/parser/pars/';

    // Формируем итоговый URL для запроса
    const finalUrl = `${endpoint}?url=${encodeURIComponent(urlInput)}`;

    // Отправляем запрос на сервер
    fetch(finalUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error('Ошибка запроса');
        }
        return response.json();
      })
      .then(data => {
        setResult(data); // Сохраняем данные в стейт
        setLoading(false);
      })
      .catch(error => {
        console.error('Ошибка при запросе:', error);
        setLoading(false);
      });
  }

  return (
    <div className="container">
      <div className="left-section">
        <h3>Парсинг URL</h3>

        {/* Поле ввода для ссылки */}
        <input
          type="text"
          placeholder="Введите URL"
          value={urlInput}
          onChange={handleUrlChange}
          className="input-field"
        />

        {/* Кнопка отправки */}
        <button onClick={handleSubmit} className="submit-button">
          {loading ? 'Загрузка...' : 'Парсить'}
        </button>

        {/* Отображение результатов */}
        {result && (
          <div className="result-section">
            <h4>Результат:</h4>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="right-section">
        <h3>Информация</h3>
        <p>
          Введите ссылку в поле выше, и мы спарсим её для вас. Результат будет
          отображён ниже в формате JSON.
        </p>
      </div>
    </div>
  );
}

export default ParserPage;
