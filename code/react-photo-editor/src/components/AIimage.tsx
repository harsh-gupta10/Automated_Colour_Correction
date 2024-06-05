import { ReactPhotoEditor } from './ReactPhotoEditor';
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled, { createGlobalStyle } from 'styled-components';
import JSZip from 'jszip';
import axios from 'axios';


// Global style to set background color
const GlobalStyle = createGlobalStyle`
  body {
    background-color: #272727;
  }
`;

const ImageContainer = styled.div`
  position: relative;
  width: 100%;
  padding-top: 100%; /* 1:1 aspect ratio */
  overflow: hidden;

  img {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
`;

const AIimage: React.FC = () => {
  const [photoUrls, setPhotoUrls] = useState<string[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<string | null>(null);
  const [file, setFile] = useState<File | undefined>();
  const [showModal, setShowModal] = useState<boolean>(false);
  const [isUploaded, setIsUploaded] = useState<boolean>(false);
  const [leftURLselected, setleftURLselected] = useState<string | null>(null);


  const [editedImageIndices, setEditedImageIndices] = useState<number[]>([]);
  const [selectedPhotoPaths, setSelectedPhotoPaths] = useState<string[]>([]);


  const [urlToFileMap, setUrlToFileMap] = useState({});
  const [urlToFileMapRev, setUrlToFileMapRev] = useState({});

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const folder = acceptedFiles[0].name;
    const urls = acceptedFiles.map((file) => URL.createObjectURL(file));
    setPhotoUrls(urls);
    setIsUploaded(true);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleFolderSelect = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.multiple = true;

    input.addEventListener('change', (event) => {
      const files = (event.target as HTMLInputElement).files;
      if (files && files.length > 0) {
        const folderName = files[0].webkitRelativePath.split('/')[0];

        const urls = [];
        const map = {};
        const mapRev = {};

        Array.from(files).forEach((file) => {
          const url = URL.createObjectURL(file);
          urls.push(url);
          map[url] = file.name;
          mapRev[file.name] = file;
        });

        setPhotoUrls(urls);
        setUrlToFileMap(map);
        setUrlToFileMapRev(mapRev);
        setIsUploaded(true);
      }
    });

    input.click();
  };

  const setFileData = (url: string) => {
    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const file = new File([blob], url, { type: 'image/jpeg' });
        setFile(file);
        // setShowModal(true);
      });
  };

  const hideModal = () => {
    setShowModal(false);
  };

  const handleSaveImage = (editedFile: File, editedUrl: string, imageSrc:string) => {
    console.log("editedFile:- ", editedFile);
    // console.log("editedUrl:- ", editedUrl);
    // console.log("imageSrc:- ", imageSrc);
    const newEditedURl = URL.createObjectURL(editedFile);


    // let fileName = editedFile.name.split('/').pop();
    // const originalUrl = urlToFileMapRev[fileName];
    // console.log("fileName:- ", fileName);
    // console.log("originalUrl:- ", originalUrl);


    const editedIndex = photoUrls.findIndex((url) => url === leftURLselected);
    console.log("editedIndex:- ", editedIndex);

    const updatedPhotoUrls = [...photoUrls];
    updatedPhotoUrls[editedIndex] = newEditedURl;
    setPhotoUrls(updatedPhotoUrls);

    let fileName = urlToFileMap[leftURLselected]
    const updatedUrlToFileMap = { ...urlToFileMap };
    delete updatedUrlToFileMap[leftURLselected];
    updatedUrlToFileMap[newEditedURl] = fileName;
    setUrlToFileMap(updatedUrlToFileMap);

    setEditedImageIndices((prevIndices) => [...prevIndices, editedIndex]);
  };

  const editor = (url: string) => {
    setSelectedPhoto(url);
    setFileData(url);
    if (file) {
      setShowModal(true);
    }
  };

  const predictImage = async (file) => {
    const formData = new FormData();
    formData.append('files', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const predictedImageData = response.data;
      return predictedImageData;
    } catch (error) {
      console.error('Error predicting image:', error);
      return null;
    }
  };

  const showModalHandler = async (url) => {
    console.log("url:- ", url);

    const imageName = urlToFileMap[url];
    const file = urlToFileMapRev[imageName];

    const predictedImageData = await predictImage(file);

    if (predictedImageData) {
      // setSelectedPhoto(url);
      // setFileData(url);
      setSelectedPhotoPaths(predictedImageData);
      setleftURLselected(url)
    }
  };

  const handleDownloadZip = async () => {
    setIsUploaded(false);
    const zip = new JSZip();
    const folder = zip.folder('Editedphotos');

    for (let i = 0; i < photoUrls.length; i++) {
      const url = photoUrls[i];
      const response = await fetch(url);
      const blob = await response.blob();
      const fileName = urlToFileMap[url] || `photo_${i}.jpg`; // Use the original file name if available, or fallback to photo_${i}.jpg
      const file = new File([blob], fileName, { type: 'image/jpeg' });
      folder.file(file.name, blob, { binary: true });
    }

    const content = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(content);
    link.download = `Editedphotos.zip`;
    link.click();
    setSelectedPhotoPaths([])
    setEditedImageIndices([])
  };

  return (
    <div className="container">
      {!isUploaded && (
        <div className="Hello mx-auto p-9">
          <div {...getRootProps()} className="border-2 border-dashed p-4 mb-4">
            <input {...getInputProps()} />
            {isDragActive ? (
              <p style={{ color: '#161616' }}>Drop the folder here...</p>
            ) : (
              <p style={{ color: '#161616' }}>Drag and drop a folder here, or click to select a folder</p>
            )}
          </div>
          <button
            onClick={handleFolderSelect}
            className="bg-[#3183D3] text-white px-4 py-2 rounded mb-4"
          >
            Select Folder
          </button>
        </div>
      )}
      {isUploaded && (
        <div className="flex" style={{ height: '100vh' }}>
          <div className="w-3/5 bg-[#272727] overflow-hidden">
            <div className="uploaded_photos h-full overflow-y-auto">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-1">
                {photoUrls.map((url, index) => (
                  <div key={index} className="relative overflow-hidden">
                    <img
                      src={url}
                      alt={`Photo ${index + 1}`}
                      className={`w-full h-auto cursor-pointer transition-transform duration-300 transform hover:scale-105 ${editedImageIndices.includes(index) ? 'border-4 border-green-500' : ''
                        }`}
                      onClick={() => showModalHandler(url)}
                    />
                    <div className="absolute inset-0 bg-black opacity-0 hover:opacity-50 transition-opacity duration-300 pointer-events-none"></div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="w-2/5 bg-[#161616] overflow-hidden">
            <div className="filters h-full overflow-y-auto">
              {selectedPhotoPaths.length > 0 && (
                <div className="pl-4">
                  <div className="grid grid-cols-2 gap-4">
                    {selectedPhotoPaths.map((imageData, index) => (
                      <div key={index} className="relative overflow-hidden">
                        <img
                          src={`data:image/jpeg;base64,${imageData}`}
                          alt={`Predicted ${index + 1}`}
                          className="w-full h-auto cursor-pointer transition-transform duration-300 transform hover:scale-105"
                          onClick={() => editor(`data:image/jpeg;base64,${imageData}`)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {selectedPhoto && (
                <ReactPhotoEditor
                  open={showModal}
                  onClose={hideModal}
                  file={file}
                  allowFlip={true}
                  allowRotate={true}
                  allowZoom={true}
                  onSaveImage={(editedFile, editedUrl, imageSrc) => handleSaveImage(editedFile, editedUrl, imageSrc)}
                  downloadOnSave={false}
                />
              )}
            </div>
          </div>
        </div>
      )}
      {isUploaded && (
        <button
          onClick={handleDownloadZip}
          className="bg-[#3183D3] fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded"
        >
          Download Zip
        </button>
      )}
    </div>
  );
};

export default AIimage;