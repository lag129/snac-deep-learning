import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report

    print(tf.config.list_physical_devices('GPU'))
    tf.random.set_seed(42)
    np.random.seed(42)
    return Path, classification_report, np, plt, tf


@app.cell
def _(Path):
    DATA_DIR = Path("data_processed")
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    return BATCH_SIZE, DATA_DIR, EPOCHS, IMAGE_SIZE, LEARNING_RATE


@app.cell
def _(BATCH_SIZE, DATA_DIR, IMAGE_SIZE, tf):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_size = len(val_test_ds) // 2
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)

    CLASS_NAMES = train_ds.class_names
    print(f"Classes: {CLASS_NAMES}")
    print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")
    return CLASS_NAMES, test_ds, train_ds, val_ds


@app.cell
def _(tf):
    factor = 0.2

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor),
            tf.keras.layers.RandomZoom(factor),
            tf.keras.layers.RandomContrast(factor),
            tf.keras.layers.RandomTranslation(factor, factor),
        ]
    )
    return (augmentation,)


@app.cell
def _(test_ds, tf, train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_1 = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds_1 = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds_1 = test_ds.prefetch(buffer_size=AUTOTUNE)
    return test_ds_1, train_ds_1, val_ds_1


@app.cell
def _(CLASS_NAMES, IMAGE_SIZE, augmentation, tf):
    base_model = tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return (model,)


@app.cell
def _(LEARNING_RATE, model, tf):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return


@app.cell
def _(EPOCHS, model, train_ds_1, val_ds_1):
    history = model.fit(train_ds_1, validation_data=val_ds_1, epochs=EPOCHS)
    return (history,)


@app.cell
def _(history, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(CLASS_NAMES, classification_report, model, np, test_ds_1):
    y_true = []
    y_pred = []
    for (images, labels) in test_ds_1:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return


@app.cell
def _(model):
    model.save('snack_classifier_epoch50.keras')
    print(f"Model saved")
    return


@app.cell
def _(CLASS_NAMES, IMAGE_SIZE, model, np, plt, tf):
    def predict_image(image_path):
        img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence:.1%})")
        plt.axis("off")
        plt.show()

        return predicted_class, confidence

    # 使い方:
    # predict_image("path/to/your/snack_image.jpg")
    return


if __name__ == "__main__":
    app.run()
