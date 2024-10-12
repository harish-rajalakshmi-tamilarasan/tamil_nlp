import streamlit as st
from load_models import generate_rnn, generate_lstm, generate_gru

st.title("Tamil Name Generator")

tamil_chars = [
    '.', 'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 
    'க', 'ச','ஞ', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ'
]
if 'selected_char' not in st.session_state:
    st.session_state.selected_char = '.' 

# st.write("Choose a starting Tamil character:")
# char_columns = st.columns(len(tamil_chars))

# for idx, char in enumerate(tamil_chars):
#     if char_columns[idx].button(char):
#         st.session_state.selected_char = char  

st.write("Choose a starting Tamil character:")
button_columns = st.columns(5)  # Number of buttons in each row

# Create buttons for Tamil characters
for idx, char in enumerate(tamil_chars):
    col_idx = idx % len(button_columns)  # Calculate column index
    if button_columns[col_idx].button(char):
        st.session_state.selected_char = char

if st.session_state.selected_char:
    st.write(f"Chosen starting character: {st.session_state.selected_char}")
else:
    st.write("Please select a character.")

col1, col2, col3 = st.columns(3)
if col1.button("Generate with RNN"):
    generated_names = generate_rnn(start_str=st.session_state.selected_char, iterations=20)
    st.write("### Generated Names (RNN):")
    for name in generated_names:
        st.write(name)

if col2.button("Generate with LSTM"):
    generated_names = generate_lstm(start_str=st.session_state.selected_char, iterations=20)
    st.write("### Generated Names (LSTM):")
    for name in generated_names:
        st.write(name)

if col3.button("Generate with GRU"):
    generated_names = generate_gru(start_str=st.session_state.selected_char, iterations=20)
    st.write("### Generated Names (GRU):")
    for name in generated_names:
        st.write(name)