import { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [searchMethod, setSearchMethod] = useState('');
  const [distanceType, setDistanceType] = useState('cosine');

  // Функция для загрузки изображения с устройства
  function handleImageUpload(event) {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(uploadedFile);
    }
  }

  // Функция для выполнения поиска
  function searchData() {
    if (!searchMethod) {
      alert('Выберите метод поиска');
      return;
    }

    let url = 'http://176.124.215.11:8000/api/search_serv/search/';
    const params = new URLSearchParams();

    params.append('search_by', searchMethod);
    params.append('distance_type', distanceType);

    // Формируем итоговый URL
    const finalUrl = `${url}?${params.toString()}`;

    // Логируем итоговый URL (можно заменить на запрос на сервер)
    console.log('Сформированный URL:', finalUrl);

    // Запрос на сервер
    fetch(finalUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(response.status);
        }
        return response.json();
      })
      .then(data => {
        setMetadata(data); // Сохраняем полученные данные
      })
      .catch(error => {
        console.error('Ошибка при поиске данных:', error);
      });
  }

  // Функция для изменения типа поиска
  function handleSearchMethodChange(event) {
    setSearchMethod(event.target.value);  // Обновляем метод поиска
  }

  // Функция для изменения типа расстояния
  function handleDistanceTypeChange(event) {
    setDistanceType(event.target.value);  // Обновляем тип расстояния
  }

  return (
    <div>
      {/* Форма для загрузки изображения с устройства */}
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      {/* Отображение загруженного изображения */}
      {image && <img src={image} alt="Uploaded" className="uploaded-image" />}

      {/* Выбор метода поиска (выпадающий список) */}
      <div>
        <label htmlFor="searchMethod">Метод поиска:</label>
        <select
          id="searchMethod"
          value={searchMethod}
          onChange={handleSearchMethodChange}
        >
          <option value="class">Class_id</option>
          <option value="one_shot_embedding">One_shot_embedding</option>
        </select>
      </div>

      {/* Выбор типа расстояния (выпадающий список) */}
      <div>
        <label htmlFor="distanceType">Тип расстояния:</label>
        <select
          id="distanceType"
          value={distanceType}
          onChange={handleDistanceTypeChange}
        >
          <option value="cosine">Cosine</option>
          <option value="euclidean">Euclidean</option>
          <option value="manhattan">Manhattan</option>
        </select>
      </div>

      {/* Кнопка для выполнения поиска */}
      <button onClick={searchData}>Найти</button>

      {/* Отображение метаданных */}
      {metadata && (
        <div>
          <pre>{JSON.stringify(metadata, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
