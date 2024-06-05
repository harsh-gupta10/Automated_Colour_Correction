
import { ReactPhotoEditor } from './ReactPhotoEditor'
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';
import JSZip, { files } from 'jszip';
// import  PhotoFilterApplier  from './PhotoFilterApplier'

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


const App: React.FC = () => {
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [photoUrls, setPhotoUrls] = useState<string[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<string | null>(null);
  const [file, setFile] = useState<File | undefined>()
  const [showModal, setShowModal] = useState<boolean>(false)
  const [isUploaded, setIsUploaded] = useState<boolean>(false);
  const [editedImageUrl, setEditedImageUrl] = useState<string | null>(null);
  const [editedImageIndex, setEditedImageIndex] = useState<number | null>(null);
  const [editedImageIndices, setEditedImageIndices] = useState<number[]>([]);
  const [selectedPhotoPaths, setSelectedPhotoPaths] = useState<string[]>([]);
  const [urlToFileMap, setUrlToFileMap] = useState({});


  const onDrop = useCallback((acceptedFiles: File[]) => {
    const folder = acceptedFiles[0].name;
    setSelectedFolder(folder);
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
        setSelectedFolder(folderName);

        const urls = [];
        const map = {};

        Array.from(files).forEach(file => {
          const url = URL.createObjectURL(file);
          urls.push(url);
          map[url] = file.name; // Map URL to file name
        });

        setPhotoUrls(urls);
        setUrlToFileMap(map); // Store the mapping
        console.log(map);

        setIsUploaded(true);
      }
    });

    input.click();
  };
  // const setFileData = (url: string) => {
  //   fetch(url)
  //     .then((response) => response.blob())
  //     .then((blob) => {
  //       const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });
  //       setFile(file);
  //     });
  // };

  const setFileData = (url: string) => {
    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });
        setFile(file);
        setShowModal(true);
      });
  };

  const hideModal = () => {
    setShowModal(false)
  }
  const handleSaveImage = (editedFile: File, editedUrl: string) => {
    setFile(editedFile);
    setEditedImageUrl(editedUrl);
    const editedIndex = photoUrls.findIndex((url) => url === selectedPhoto);
    setEditedImageIndex(editedIndex);

    // Update the photoUrls state with the edited image URL
    const updatedPhotoUrls = [...photoUrls];
    updatedPhotoUrls[editedIndex] = editedUrl;
    setPhotoUrls(updatedPhotoUrls);

    // Add the edited image index to the editedImageIndices state
    setEditedImageIndices((prevIndices) => [...prevIndices, editedIndex]);
  };
  const editor = (url: string) => {
    setSelectedPhoto(url);
    setFileData(url);
    if (file) {
      setShowModal(true)
      console.log("GOt the setted file");
    }
  };
  const showModalHandler = (url: string) => {
    // console.log(url);
    console.log(urlToFileMap[url]);

    // const imageName = url.split('/').pop();
    const imageName = urlToFileMap[url];
    console.log(imageName);

    const selectedPhotoPaths = [
      `../../photos/result/ANN/A/${imageName}`,
      `../../photos/result/ANN/B/${imageName}`,
      `../../photos/result/LR/A/${imageName}`,
      `../../photos/result/LR/B/${imageName}`,
      `../../photos/result/RSN/A/${imageName}`,
      `../../photos/result/RSN/B/${imageName}`,
    ];
    console.log(selectedPhotoPaths);
    setSelectedPhotoPaths(selectedPhotoPaths);
  };




  const handleDownloadZip = async () => {
    const zip = new JSZip();
    const folder = zip.folder('Editedphotos');

    for (let i = 0; i < photoUrls.length; i++) {
      const url = photoUrls[i];
      const response = await fetch(url);
      const blob = await response.blob();
      const file = new File([blob], `photo_${i}.jpg`, { type: 'image/jpeg' });
      folder.file(file.name, blob, { binary: true });
    }

    const content = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(content);
    link.download = `${'Editedphotos'}.zip`;
    link.click();
  };

  return (
    <div className="container">
      {!isUploaded && (
        <div className="Hello mx-auto p-9">
          <div {...getRootProps()} className="border-2 border-dashed p-4 mb-4">
            <input {...getInputProps()} />
            {isDragActive ? (
              <p>Drop the folder here...</p>
            ) : (
              <p>Drag and drop a folder here, or click to select a folder</p>
            )}
          </div>
          <button
            onClick={handleFolderSelect}
            className="bg-blue-500 text-white px-4 py-2 rounded mb-4"
          >
            Select Folder
          </button>
        </div>
      )}
      {isUploaded && (
        <div className="flex">
          <div className="w-1/3 bg-blue-900" style={{ minHeight: '100%' }}>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
              {photoUrls.map((url, index) => (
                <div key={index} className="relative overflow-hidden">
                  <img
                    src={url}
                    alt={`Photo ${index + 1}`}
                    className={`w-full h-auto cursor-pointer transition-transform duration-300 transform hover:scale-105 ${editedImageIndices.includes(index) ? 'border-4 border-green-500' : ''}`}
                    onClick={() => showModalHandler(url)}
                  />
                  <div className="absolute inset-0 bg-black opacity-0 hover:opacity-50 transition-opacity duration-300 pointer-events-none"></div>
                </div>
              ))}
            </div>
          </div>
          <div className="w-2/3">
            {selectedPhotoPaths.length > 0 && (
              <div className="pl-4">
                <div className="grid grid-cols-2 gap-4">
                  {selectedPhotoPaths.map((path, index) => (
                    <div key={index} className="relative overflow-hidden">
                      <img
                        src={path}
                        alt={`Photo ${index + 1}`}
                        className="w-full h-auto cursor-pointer transition-transform duration-300 transform hover:scale-105"
                        onClick={() => editor(path)}
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
                onSaveImage={(editedFile, editedUrl) => handleSaveImage(editedFile, editedUrl)}
                downloadOnSave={false}
              />
            )}
          </div>
        </div>
      )}
      {isUploaded && (
        <button
          onClick={handleDownloadZip}
          className="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded"
        >
          Download Zip
        </button>
      )}
    </div>

  );
};



export default App;