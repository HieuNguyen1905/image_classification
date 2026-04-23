import { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Vì đã cấu hình proxy trong vite.config.js, có thể gọi trực tiếp /predict 
      // (hoặc gọi http://localhost:8000/predict nếu không có proxy)
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Có lỗi xảy ra khi dự đoán.');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message || 'Lỗi kết nối đến máy chủ.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Phân Loại Động Vật 🐾</h1>
        <p>Hệ thống nhận diện Mèo, Vịt và Gấu Trúc</p>
      </header>

      <main className="main-content">
        <form onSubmit={handleSubmit} className="upload-section">
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-upload"
              accept="image/*"
              onChange={handleFileChange}
              className="file-input"
            />
            <label htmlFor="file-upload" className="file-label">
              {selectedFile ? 'Chọn ảnh khác' : 'Tải ảnh lên'}
            </label>
          </div>

          <button 
            type="submit" 
            className={`submit-btn ${!selectedFile || loading ? 'disabled' : ''}`}
            disabled={!selectedFile || loading}
          >
            {loading ? 'Đang phân tích...' : 'Phân loại'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        <div className="result-section">
          <div className="image-preview">
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" />
            ) : (
              <div className="placeholder">Chưa có ảnh nào được chọn</div>
            )}
          </div>

          {prediction && (
            <div className="prediction-box">
              <h2>Kết quả dự đoán</h2>
              <div className="result-item">
                <span className="label">Loài: </span>
                <span className="value result-class">
                  {prediction.prediction === 'cat' ? '🐱 Mèo' : 
                   prediction.prediction === 'duck' ? '🦆 Vịt' : 
                   prediction.prediction === 'panda' ? '🐼 Gấu Trúc' : 
                   prediction.prediction}
                </span>
              </div>
              <div className="result-item">
                <span className="label">Độ tin cậy: </span>
                <span className="value result-confidence">
                  {(prediction.confidence * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;