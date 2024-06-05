import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

function ImageUploader() {
  const [files, setFiles] = useState([]);
  const [predictedImage, setPredictedImage] = useState(null);
  const [predictedImageData, setPredictedImageData] = useState(null);


  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      setFiles(acceptedFiles.map((file) => Object.assign(file, { preview: URL.createObjectURL(file) })));
    },
  });

  const handleImageClick = async (file) => {
    const formData = new FormData();
    formData.append('files', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setPredictedImage(`data:image/jpeg;base64,${response.data[0]}`);
      setPredictedImageData(response.data[0]);
    } catch (error) {
      console.error('Error uploading files:', error);
    }
  };

  const renderPredictedImage = () => {
    if (!predictedImageData) return null;

    const base64Data = `data:image/jpeg;base64,${predictedImageData}`;
    const binaryData = atob(base64Data.split(',')[1]);
    const arrayBuffer = new ArrayBuffer(binaryData.length);
    const uint8Array = new Uint8Array(arrayBuffer);

    for (let i = 0; i < binaryData.length; i++) {
      uint8Array[i] = binaryData.charCodeAt(i);
    }

    const blob = new Blob([uint8Array], { type: 'image/jpeg' });
    const imageUrl = URL.createObjectURL(blob);

    return <img src={imageUrl} alt="Predicted" style={{ maxWidth: '100%', maxHeight: '100%' }} />;
  };

  return (
    <div style={{ display: 'flex' }}>
      <div style={{ flex: 1, padding: '1rem' }}>
        <div {...getRootProps({ style: { border: '1px dashed gray', padding: '1rem' } })}>
          <input {...getInputProps()} />
          <p>Drag and drop files here, or click to select files</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
          {files.map((file) => (
            <div key={file.name} style={{ cursor: 'pointer' }} onClick={() => handleImageClick(file)}>
              <img src={file.preview} alt={file.name} style={{ width: '100%', height: 'auto' }} />
            </div>
          ))}
        </div>
      </div>
      <div style={{ flex: 1, padding: '1rem' }}>
        {predictedImage && <img src={predictedImage} alt="Predicted" style={{ maxWidth: '100%', maxHeight: '100%' }} />}
      </div>
      <div style={{ flex: 1, padding: '1rem' }}>
        {renderPredictedImage()}
      </div>

    </div>
  );
}

export default ImageUploader