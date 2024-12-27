import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

def show_demonstration():
    st.title("Demonstration")
    st.write("Upload a chest X-ray image to see the model's prediction.")

    def preprocess_image(image, model_name):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)

        elif image.shape[-1] == 4:
            image = image[..., :3]

        if model_name == 'CNN 1.1':
            image = tf.image.resize(image, [256, 256])
            image = tf.image.rgb_to_grayscale(image)
            image = tf.cast(image, tf.float32) / 255.0
        elif model_name == 'TL Model (EfficientNetB1)':
            image = tf.image.resize(image, [240, 240])
            image = preprocess_input(image)

        return image

    def get_gradcam_heatmap(model, img_array, last_conv_layer_name, model_name):
        if model_name == 'CNN 1.1':
            grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(last_conv_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_array)
                
                top_pred_index = tf.argmax(preds[0])
                top_class_channel = preds[:, top_pred_index]

            grads = tape.gradient(top_class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0] 

            heatmap = conv_outputs * pooled_grads

            heatmap = tf.reduce_sum(heatmap, axis=-1)

        elif model_name == 'TL Model (EfficientNetB1)':
            grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

            # Record operations for gradient calculation
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                predictions = tf.squeeze(predictions)

                top_pred_index = tf.argmax(predictions)

                loss = predictions[top_pred_index]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0].numpy()
            pooled_grads = pooled_grads.numpy()

            if conv_outputs.ndim == 3 and pooled_grads.ndim == 1:
                heatmap = np.tensordot(conv_outputs, pooled_grads, axes=(2, 0))
            else:
                st.write("Error: Dimensions of conv_outputs and pooled_grads do not match for multiplication.")

            heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    # Function to display Grad-CAM
    def display_gradcam(img, heatmap, model_name, alpha=0.4):
        heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0])))
    
        colormap = plt.get_cmap('jet')
        normalized_heatmap = heatmap / np.max(heatmap)

        if model_name == 'CNN 1.1':
            colored_heatmap = colormap(normalized_heatmap)[..., :3] * 255
            img_pil = Image.fromarray((np.squeeze(img) * 255).astype(np.uint8)).convert("RGB")
            heatmap_image = Image.fromarray(colored_heatmap.astype(np.uint8))

        elif model_name == 'TL Model (EfficientNetB1)':
            colored_heatmap = colormap(normalized_heatmap)
            colored_heatmap = (colored_heatmap[..., :3] * 255).astype(np.uint8)

            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            heatmap_image = Image.fromarray(colored_heatmap)
               
        superimposed_img = Image.blend(img_pil, heatmap_image, alpha)
        return np.array(superimposed_img) / 255

    # Function to load models
    @st.cache_resource
    def load_model(model_name):
        if model_name == 'CNN 1.1':
            model = tf.keras.models.load_model('Models/cnn_1.1.keras')
        elif model_name == 'TL Model (EfficientNetB1)':
            model = tf.keras.models.load_model('Models/tl_EfficientNetB1.keras')
        return model
        
    # Model selection
    model_name = st.selectbox("Choose a model", ("CNN 1.1", "TL Model (EfficientNetB1)"))
    model = load_model(model_name)

    # Set the last convolutional layer name based on the model
    if model_name == 'CNN 1.1':
        last_conv_layer_name = 'conv2d_4'
    elif model_name == 'TL Model (EfficientNetB1)':
        last_conv_layer_name = 'top_conv'

    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_image = preprocess_image(np.array(image), model_name)
        input_image = np.expand_dims(input_image, axis=0)

        predictions = model.predict(input_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        prob = np.max(predictions)

        class_labels = ["Normal", "COVID", "Lung Opacity", "Viral Pneumonia"]
        predicted_label = class_labels[predicted_class]

        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Prediction Probability: {prob:.4f}")

        heatmap = get_gradcam_heatmap(model, input_image, last_conv_layer_name, model_name)

        if model_name == 'CNN 1.1':
            gradcam_image = display_gradcam(np.squeeze(input_image), heatmap, model_name)
            st.image(gradcam_image, caption="Grad-CAM", use_column_width=True)
        elif model_name == 'TL Model (EfficientNetB1)':
            input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
            gradcam_image = display_gradcam(np.squeeze(input_image), heatmap, model_name)
            st.image(gradcam_image, caption="Grad-CAM", use_column_width=True)

        true_label = st.text_input("Enter the true class label if available:")
        if true_label:
            st.write(f"True Class: {true_label}")