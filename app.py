import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('potatoes.h5')


def classify_disease(image):
    image = tf.image.resize(image, (256, 256))  # Resize to (256, 256)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = image.reshape((1, 256, 256, 3))  # Update the shape
    result = model.predict(image)
    return result



# Set the title and description
st.title('🥔🔍✨ AGGRINATION 📸🌱🦠')
st.write('This web app classifies potato diseases into three categories: Early Blight, Late Blight, and Healthy.')

# Upload the image
uploaded_image = st.file_uploader("Upload a potato image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Classify"):
        result = classify_disease(image)



        CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
        predicted_class = CLASS_NAMES[np.argmax(result[0])]

        st.success(f'Predicted Class: {predicted_class}')
        st.write('Prediction Probabilities:')
        if predicted_class == 'Early Blight':
            st.write("It appears that your potato plants are predicted to have Early Blight.")
            st.write('બટાકાના રોગના પ્રારંભિક ખુમારીને આપણે કેવી રીતે નિયંત્રિત કરી શકીએ?')
            st.write('યોગ્ય ગર્ભાધાન, સિંચાઈ અને અન્ય જંતુઓના વ્યવસ્થાપન સહિત શ્રેષ્ઠ વૃદ્ધિની સ્થિતિ જાળવી રાખીને પ્રારંભિક ખુમારીને ઘટાડી શકાય છે. પછીથી પાકતી, લાંબી મોસમની જાતો ઉગાડો. ફૂગનાશકનો ઉપયોગ ત્યારે જ વાજબી છે જ્યારે રોગ આર્થિક નુકસાન પહોંચાડવા માટે પૂરતો વહેલો શરૂ કરવામાં આવે.')
            st.write("ફૂગનાશક સ્પ્રે પ્રારંભિક ખુમારી અને અન્ય પાંદડાના ફોલ્લીઓને નિયંત્રિત કરવા માટે અસરકારક છે. ક્લોરોથેલોનિલ (0.20%), મેન્કોઝેબ (0.20%) અથવા પ્રોપીનેબ (0.20%) સાથે પાક પર છંટકાવ કરવાથી આ રોગોની કાળજી લઈ શકાય છે. ખાતરોની ભલામણ કરેલ માત્રા ખાસ કરીને નાઈટ્રોજનનો ઉપયોગ કરો.")
        elif predicted_class == 'Late Blight':
            st.write("It appears that your potato plants are predicted to have Late Blight.")
            st.write('What is the solution for late blight of potato?')
            st.write("કેલ્શિયમ પોષક તત્ત્વો સાથે છાંટવામાં આવતા બટાકાના છોડમાં મોડા બ્લાઈટ રોગની ઓછી તીવ્રતા અને કંદની વધેલી ઉપજ છોડની પેશીઓમાં કેલ્શિયમના વધુ સંચયને કારણે હોઈ શકે છે.")
            st.write('લેટ બ્લાઈટ/બટાટા/ખેતી: જંતુ વ્યવસ્થાપન...કલ પાઈલ્સ અને સ્વયંસેવક બટાકાને નાબૂદ કરીને, યોગ્ય લણણી અને સંગ્રહ પદ્ધતિઓનો ઉપયોગ કરીને અને જ્યારે જરૂરી હોય ત્યારે ફૂગનાશકનો ઉપયોગ કરીને લેટ બ્લાઈટને નિયંત્રિત કરવામાં આવે છે. દરરોજ પર્ણસમૂહને સૂકવવા માટે હવાનું ડ્રેનેજ મહત્વપૂર્ણ છે.')
        else:
            st.write(
                "Your potato plants appear to be healthy. Regularly fertilize them with a balanced fertilizer to maintain their health and vigor.")

# Footer
st.write('🌟 Thank you for using our website! 😊👍')
