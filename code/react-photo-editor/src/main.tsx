import ReactDOM from 'react-dom/client'
import App from './App'
import FolderUpload from './components/FolderUpload'
import ImageUploader from './components/ImageUploader'
import AIimage from './components/AIimage'


ReactDOM.createRoot(document.getElementById('root')!).render(
  // <React.StrictMode>
  <div>
    {/* <App /> */}
    {/* <FolderUpload /> */}
    {/* <ImageUploader/> */}
    <AIimage/>
  </div>
  // </React.StrictMode>,
)
