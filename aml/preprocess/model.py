import tensorflow as tf
from google.cloud import storage
import glob
from keras.applications import vgg16
from keras import layers, Sequential
import subprocess
import numpy as np
from PIL import Image
from aml.params import *

def define_model():
    vgg16_basemodel=vgg16.VGG16(include_top=False,
                  input_shape=(144,144,3),
                  weights="imagenet"
                  )
    vgg16_basemodel.trainable=False
    model=Sequential()
    model.add(vgg16_basemodel)
    model.add(layers.Flatten())
    model.add(layers.Dense(250, activation="relu"))
    model.add(layers.Dense(5,activation="softmax"))
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use categorical_crossentropy if one-hot encoded
    metrics=['accuracy']
    )
    return model
def create_model(model):
    images_dir=f"gs://{BUCKET_NAME}/aml_data"
    class_folders = subprocess.run(['gsutil', 'ls', images_dir], stdout=subprocess.PIPE)
    class_folders = class_folders.stdout.decode('utf-8').splitlines()
    class_names=sorted([folder.split("/")[-2] for folder in class_folders])
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    filepaths = []
    labels = []
    for class_name in class_names:
        class_path = os.path.join(images_dir, class_name)
        result = subprocess.run(['gsutil', 'ls', f'{class_path}/*.tif'], stdout=subprocess.PIPE)
        tif_files = result.stdout.decode('utf-8').splitlines()
        for fname in tif_files:
                filepaths.append(fname)
                labels.append(class_to_index[class_name])
    # for reproducible split
    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(filepaths))
    rng.shuffle(indices)
    file_paths = [filepaths[i] for i in indices]
    class_labels = [labels[i] for i in indices]
    # Compute split sizes
    n = len(file_paths)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_files, train_labels = file_paths[:train_end], class_labels[:train_end]
    val_files, val_labels     = file_paths[train_end:val_end], class_labels[train_end:val_end]
    test_files, test_labels   = file_paths[val_end:], class_labels[val_end:]
    def load_tiff(path):
        path = path.numpy().decode('utf-8')
        with tf.io.gfile.GFile(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')        # Ensure 3 channels
            img = img.resize((144, 144))    # Resize as needed
            image_array = np.array(img).astype(np.float32) / 255.0
            return image_array

    def tf_load_tiff(path, label):
        img = tf.py_function(load_tiff, [path], Tout=tf.float32)
        img.set_shape([144, 144, 3])
        return img, label

    def make_dataset(files, labels, shuffle=True, batch_size=32):
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)
        ds = ds.map(tf_load_tiff, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    batch_size = 32
    train_ds = make_dataset(train_files, train_labels, shuffle=True, batch_size=batch_size)
    val_ds   = make_dataset(val_files, val_labels, shuffle=False, batch_size=batch_size)
    test_ds  = make_dataset(test_files, test_labels, shuffle=False, batch_size=batch_size)
    model=create_model()
    model.fit(train_ds, epochs=1, batch_size=32,verbose=1, validation_data=val_ds)
    return model

def load_model_gcs() -> tf.keras.Model:
    """
    Return a saved model:
    - from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    Return None (but do not Raise) if no model is found

    """

    print( "Load latest model from GCS..." )
    bucket_name = BUCKET_NAME
    models_dir = "models"
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=models_dir))
    model_blobs = [blob for blob in blobs if blob.name.endswith('.keras')]

    try:
        latest_blob = max(model_blobs, key=lambda b: b.updated)
        gcs_model_path = f"gs://{bucket_name}/{latest_blob.name}"
        model = tf.keras.models.load_model(gcs_model_path)
        return model
    except:
        print(f"\n❌ No model found in GCS bucket")
        return None

def load_model_local() -> tf.keras.Model:
    """
    Return a saved model:
    - from local (most recent one)
    Return None (but do not Raise) if no model is found
    """
    model_path=os.path.join(os.path.dirname(__file__), "..", "models")
    print(model_path)
    model_files = glob.glob(os.path.join(model_path, '*.h5'))
    try:
        print(model_files)
        latest_model = max(model_files, key=os.path.getmtime)
        model=tf.keras.models.load_model(latest_model)
        return model
    except:
        return None
