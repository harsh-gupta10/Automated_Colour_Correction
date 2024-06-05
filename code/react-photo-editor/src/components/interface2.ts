export interface ReactPhotoEditorProps {
  /**
   * The input image files to be edited.
   */
  files: File[];

  /**
   * Whether to allow color editing options.
   * @default true
   */
  allowColorEditing?: boolean;

  /**
   * Whether to allow rotation of the image.
   * @default true
   */
  allowRotate?: boolean;

  /**
   * Whether to allow flipping (horizontal/vertical) of the image.
   * @default true
   */
  allowFlip?: boolean;

  /**
   * Whether to allow zooming of the image.
   * @default true
   */
  allowZoom?: boolean;

  /**
   * Whether the photo editor modal is open.
   * @default false
   */
  open?: boolean;

  /**
   * Function invoked when the photo editor modal is closed.
   */
  onClose?: () => void;

  /**
   * Function invoked when each edited image is saved.
   * @param editedFile - The edited image file.
   * @param editedUrl - The URL of the edited image.
   * @param imageSrc - The source URL of the original image.
   */
  onSaveImage: (editedFile: File, editedUrl: string, imageSrc: string) => void;

  /**
   * An array of HSV values corresponding to each file in the `files` array.
   */
  hsvValues: { hue: number; saturation: number; value: number }[];
}
