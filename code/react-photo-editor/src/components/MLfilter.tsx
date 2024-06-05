import { useEffect, useRef, useState, ChangeEvent } from 'react';
import { MLfilterProps } from './interface2';
import './style.css';

const modalHeaderButtonClasses = "ml-2 text-md outline-none py-1 px-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700  focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-700 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700";

export const MLfilter: React.FC<MLfilterProps> = ({
  files,
  onSaveImage,
  allowColorEditing = true,
  allowFlip = true,
  allowRotate = true,
  allowZoom = true,
  open,
  onClose,
  hsvValues,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [imageSrc, setImageSrc] = useState('');
  const [imageName, setImageName] = useState('');
  const [rotate, setRotate] = useState(0);
  const [flipHorizontal, setFlipHorizontal] = useState(false);
  const [flipVertical, setFlipVertical] = useState(false);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (files && files.length > 0) {
      const fileSrc = URL.createObjectURL(files[currentFileIndex]);
      setImageSrc(fileSrc);
      setImageName(files[currentFileIndex].name);
      return () => {
        URL.revokeObjectURL(fileSrc);
      };
    }
  }, [files, currentFileIndex, open]);

  useEffect(() => {
    applyFilter();
  }, [imageSrc, rotate, flipHorizontal, flipVertical, zoom, hsvValues]);

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
    const { hue, saturation, value } = hsvValues[currentFileIndex];
    return `hue-rotate(${hue}deg) saturate(${saturation}%) brightness(${value}%)`;
  };

  const handleRotateChange = (event: ChangeEvent<HTMLInputElement>) => {
    setRotate(parseInt(event?.target?.value));
  };

  const handleZoomIn = () => {
    setZoom((prevZoom) => prevZoom + 0.1);
  };

  const handleZoomOut = () => {
    setZoom((prevZoom) => Math.max(prevZoom - 0.1, 0.1));
  };

  const resetImage = () => {
    setRotate(0);
    setFlipHorizontal(false);
    setFlipVertical(false);
    setZoom(1);
  };

  const saveImage = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.toBlob((blob) => {
        if (blob) {
          const editedFile = new File([blob], imageName, { type: blob.type });
          const editedUrl = URL.createObjectURL(editedFile);
          onSaveImage(editedFile, editedUrl, imageSrc);
          if (currentFileIndex < files.length - 1) {
            setCurrentFileIndex((prevIndex) => prevIndex + 1);
          } else {
            if (onClose) {
              onClose();
            }
          }
        }
      });
      resetImage();
    }
  };

  const closeEditor = () => {
    resetImage();
    if (onClose) {
      onClose();
    }
  };

  return (
    <>
      {open && (
        <>
          {/* ... */}
          <div className="p-4">
            <div className="flex flex-col">
              <canvas
                className="canvas overflow-auto max-w-lg max-h-[22rem] w-full object-contain mx-auto"
                data-testid="image-editor-canvas"
                id="canvas"
                ref={canvasRef}
              />
              {/* ... */}
            </div>
          </div>
          {/* ... */}
        </>
      )}
    </>
  );
};