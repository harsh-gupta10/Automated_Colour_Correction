import { useEffect, useRef, useState, ChangeEvent } from 'react'
import { ReactPhotoEditorProps } from './interface';
import './style.css'

const modalHeaderButtonClasses = "ml-2 text-md outline-none py-1 px-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700  focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-700 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
export const ReactPhotoEditor: React.FC<ReactPhotoEditorProps> = ({
  file,
  onSaveImage,
  allowColorEditing = true,
  allowFlip = true,
  allowRotate = true,
  allowZoom = true,
  // downloadOnSave,
  open,
  onClose
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [imageSrc, setImageSrc] = useState('');
  const [imageName, setImageName] = useState('');
  const [brightnessValue, setBrightnessValue] = useState(100);
  const [hueValue, sethueValue] = useState(0);
  const [saturateValue, setSaturateValue] = useState(100);
  const [grayscaleValue, setGrayscaleValue] = useState(0);
  const [rotate, setRotate] = useState(0);
  const [flipHorizontal, setFlipHorizontal] = useState(false);
  const [flipVertical, setFlipVertical] = useState(false);
  const [zoom, setZoom] = useState(1);


  useEffect(() => {

    if (file) {
      // let fileName = file.name.split('/').pop();
      const fileSrc = URL.createObjectURL(file);
      setImageSrc(fileSrc);
      setImageName(file.name);
      // console.log(file);
      // console.log(imageSrc);
      
      return () => {
        URL.revokeObjectURL(fileSrc);
      }
    }

  }, [file, open])

  useEffect(() => {
    applyFilter();
  }, [file, imageSrc, rotate, flipHorizontal, flipVertical, zoom, brightnessValue, hueValue, saturateValue, grayscaleValue]);

  const applyFilter = () => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const image = new Image();
    image.src = imageSrc;
    image.onload = () => {
      if (canvas && context) {
        const zoomedWidth = image.width * zoom;
        const zoomedHeight = image.height * zoom;
        const translateX = (image.width - zoomedWidth) / 2;
        const translateY = (image.height - zoomedHeight) / 2;
        canvas.width = image.width;
        canvas.height = image.height;
        context.filter = getFilterString();
        context.save();
        if (rotate) {
          const centerX = canvas.width / 2;
          const centerY = canvas.height / 2;
          context.translate(centerX, centerY);
          context.rotate((rotate * Math.PI) / 180);
          context.translate(-centerX, -centerY);
        }
        if (flipHorizontal) {
          context.translate(canvas.width, 0);
          context.scale(-1, 1);
        }
        if (flipVertical) {
          context.translate(0, canvas.height);
          context.scale(1, -1);
        }
        context.translate(translateX, translateY);
        context.scale(zoom, zoom);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);

        context.restore();
      }
    };
  };

  const getFilterString = () => {
    return `hue-rotate(${hueValue}deg) saturate(${saturateValue}%) brightness(${brightnessValue}%)`;
  };

  const handleRotateChange = (event: ChangeEvent<HTMLInputElement>) => {
    setRotate(parseInt(event?.target?.value));
  };

  const handleZoomIn = () => {
    setZoom(prevZoom => prevZoom + 0.1);
  };

  const handleZoomOut = () => {
    setZoom(prevZoom => Math.max(prevZoom - 0.1, 0.1));
  };

  const resetImage = () => {
    setBrightnessValue(100);
    sethueValue(0);
    setSaturateValue(100);
    setGrayscaleValue(0);
    setRotate(0);
    setFlipHorizontal(false);
    setFlipVertical(false);
    setZoom(1);
  }

  const saveImage = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.toBlob((blob) => {
        if (blob) {
          const editedFile = new File([blob], imageName, { type: blob.type });
          const editedUrl = URL.createObjectURL(editedFile);
          onSaveImage(editedFile, editedUrl, imageSrc);
          if (onClose) {
            onClose();
          }
        }
      });
      resetImage();
    }
  };

  const closeEditor = () => {
    resetImage()
    if (onClose) {
      onClose();
    }
  }
  return (
    <>
      {open && (
        <>
          <div data-testid="photo-editor-main" className="justify-center items-center flex overflow-auto fixed inset-0">
            <div className="relative rounded-lg shadow-lg w-[40rem] max-sm:w-[22rem] bg-white h-[38rem] dark:bg-slate-800">
              <div className="flex justify-end p-2 rounded-t">
                <button
                  className={modalHeaderButtonClasses}
                  onClick={closeEditor}
                >
                  Close
                </button>
                <button
                  className={modalHeaderButtonClasses}
                  onClick={() => saveImage()}
                  data-testid="save-button"
                >
                  Save
                </button>
              </div>
              <div className="p-4">
                <div className="flex flex-col">
                  <canvas className='canvas overflow-auto max-w-lg max-h-[22rem] w-full object-contain mx-auto' data-testid="image-editor-canvas" id="canvas" ref={canvasRef} />

                  <div className='items-center flex m-1 flex-col'>

                    <div className='flex flex-col bottom-12 gap-1 mt-4 max-sm:w-72 w-11/12 absolute '>
                      {allowRotate && <div className='flex flex-row items-center'>
                        <label className="text-xs font-medium text-gray-900 dark:text-white">Rotate: </label>
                        <input
                          type='range'
                          value={rotate}
                          step='1'
                          onChange={handleRotateChange}
                          min={-180}
                          max={180}
                          className='ml-[1.8rem] w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700'
                        />
                        <input
                          type="number"
                          value={rotate}
                          onChange={handleRotateChange}
                          className='w-14 pl-1 rounded-md text-right'
                        />
                      </div>}
                      {allowColorEditing &&
                        <>
                          <div className='flex items-center'>
                            <label className="text-xs font-medium text-gray-900 dark:text-white">Hue: </label>
                            <input
                              id="default-range"
                              type="range"
                              step="1"
                              value={hueValue}
                              max={200}
                              onChange={e => sethueValue(parseInt(e.target.value))}
                              // className='ml-[1.1rem] w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700'
                              className='ml-[0.8rem] w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700'

                            />
                            <input
                              type="number"
                              value={hueValue}
                              onChange={e => sethueValue(parseInt(e.target.value))}
                              className='w-14 pl-1 rounded-md text-right'
                            />

                          </div>
                          <div className='flex flex-row justify-between items-center' data-testid="color-editing">
                            <label className="text-xs font-medium text-gray-900 dark:text-white">Brightness: </label>
                            <input
                              id="default-range"
                              type="range"
                              step="1"
                              max={200}
                              value={brightnessValue}
                              onChange={e => setBrightnessValue(parseInt(e.target.value))}
                              className='ml-2 w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700'
                            />
                            <input
                              type="number"
                              value={brightnessValue}
                              onChange={e => setBrightnessValue(parseInt(e.target.value))}
                              className='w-14 pl-1 rounded-md text-right'
                            />

                          </div>


                          <div className='flex items-center'>
                            <label className="text-xs font-medium text-gray-900 dark:text-white">Saturate: </label>
                            <input
                              id="default-range"
                              type="range"
                              step="1"
                              max={200}
                              value={saturateValue}
                              onChange={e => setSaturateValue(parseInt(e.target.value))}
                              className='ml-[1.2rem] w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700'
                            />
                            <input
                              type="number"
                              value={saturateValue}
                              onChange={e => setSaturateValue(parseInt(e.target.value))}
                              className='w-14 pl-1 rounded-md text-right'
                            />
                          </div>
                          {/* <div className='flex items-center'>
                            <label className="text-xs font-medium text-gray-900 dark:text-white">Grayscale: </label>
                            <input
                              id="grayscaleSlider"
                              type="range"
                              value={grayscaleValue}
                              max="100"
                              onChange={e => setGrayscaleValue(parseInt(e.target.value))}
                              className='ml-[0.8rem] w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer range-sm dark:bg-gray-700'
                            />
                            <input
                              type="number"
                              value={grayscaleValue}
                              onChange={e => setGrayscaleValue(parseInt(e.target.value))}
                              className='w-14 pl-1 rounded-md text-right'
                            />
                          </div> */}
                        </>
                      }
                    </div>
                  </div>
                  <div className='flex justify-center'>

                    <div className='mb-1 absolute bottom-0 mt-2'>
                      <button className='mx-1  focus:ring-4 rounded-md p-1' onClick={resetImage}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" className="lucide lucide-rotate-ccw dark:stroke-slate-200"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></svg>
                      </button>
                      {allowFlip &&
                        <div className='inline-block' data-testid="flip-btns">
                          <button className='mx-1 focus:ring-4 rounded-md p-1' onClick={() => setFlipHorizontal(!flipHorizontal)}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-flip-horizontal dark:stroke-slate-200"><path d="M8 3H5a2 2 0 0 0-2 2v14c0 1.1.9 2 2 2h3" /><path d="M16 3h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3" /><path d="M12 20v2" /><path d="M12 14v2" /><path d="M12 8v2" /><path d="M12 2v2" /></svg>
                          </button>
                          <button className='mx-1 focus:ring-4 rounded-md p-1' onClick={() => setFlipVertical(!flipVertical)}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-flip-vertical dark:stroke-slate-200"><path d="M21 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v3" /><path d="M21 16v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-3" /><path d="M4 12H2" /><path d="M10 12H8" /><path d="M16 12h-2" /><path d="M22 12h-2" /></svg>
                          </button>
                        </div>
                      }
                      {allowZoom &&
                        <div className='inline-block' data-testid="zoom-btns">
                          <button data-testid="zoom-in" className='mx-1 focus:ring-4 rounded-md p-1' onClick={handleZoomIn}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-zoom-in dark:stroke-slate-200"><circle cx="11" cy="11" r="8" /><line x1="21" x2="16.65" y1="21" y2="16.65" /><line x1="11" x2="11" y1="8" y2="14" /><line x1="8" x2="14" y1="11" y2="11" /></svg>
                          </button>
                          <button data-testid="zoom-out" className='mx-1 focus:ring-4 rounded-md p-1' onClick={handleZoomOut}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-zoom-out dark:stroke-slate-200"><circle cx="11" cy="11" r="8" /><line x1="21" x2="16.65" y1="21" y2="16.65" /><line x1="8" x2="14" y1="11" y2="11" /></svg>
                          </button>
                        </div>
                      }
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          {/* <div className="opacity-75 fixed inset-0 z-40 bg-black"></div> */}
        </>
      )}
    </>
  );
}

