import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import re

# Define common quantity/packaging keywords to remove (lowercase)
QUANTITY_KEYWORDS = [
    'case', 'cases', 'box', 'boxes', 'bag', 'bags', 'pound', 'pounds', 'lb', 'lbs', 'lbst',
    'pack', 'packs', 'package', 'packages', 'unit', 'units', 'each', 'ready', 'cut', 'chopped',
    'ct', 'ctn', 'g', 'kg', 'ml', 'l', 'gallon', 'gallons', 'oz', 'ounce', 'ounces',
    'kilo', 'kilos', 'gram', 'grams', 'liter', 'liters', 'bunch', 'bunches', 'piece', 'pieces',
    'bottle', 'bottles', 'jar', 'jars', 'can', 'cans', 'tube', 'tubes', 'roll', 'rolls',
    'big', 'small', 'large', 'medium', 'jumbo', 'extra', 'light', 'heavy', 'thick', 'thin', 'fresh', 'dried', 'frozen',
    'new', 'old', 'bulk', 'loose', 'tight', 'fine', 'coarse', 'ground', 'whole', 'half', 'quarter',
    'single', 'double', 'triple', 'jumbo', 'xl', 'l', 'm', 's', 'xs', 'xxl', 'xxxl'
]

# --- 1. Load Your Data ---
DATA_FILE = 'customer_orders.csv'

@st.cache_data
def load_order_data():
    """Load and cache the order data"""
    if not os.path.exists(DATA_FILE):
        st.error(f"Error: '{DATA_FILE}' not found. Please ensure the CSV file with your customer order data is in the same folder as 'app.py'.")
        st.stop()

    try:
        df = pd.read_csv(DATA_FILE)
        # Ensure required columns exist
        required_columns = ['customer_id', 'customer_name', 'customer_phrase', 'internal_sku']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns in CSV: {missing_columns}")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading data from '{DATA_FILE}'. Please check its format and content. Error: {e}")
        st.stop()

# --- 2. Load NLP Model ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("SpaCy model 'en_core_web_sm' not found. Attempting to download (this might take a moment)...")
        try:
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to download or load SpaCy model. Please try running 'python -m spacy download en_core_web_sm' in your terminal manually. Error: {e}")
            st.stop()

# --- 3. Text Normalization Function ---
def normalize_text(text, nlp_model):
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\b\d+\b', '', text) # Remove whole numbers
    text = re.sub(r'\b[a-z]\b', '', text) # Remove single letters

    doc = nlp_model(text)

    tokens = [
        token.lemma_ for token in doc
        if not token.is_punct and
           not token.is_stop and
           token.text not in QUANTITY_KEYWORDS and
           len(token.text) > 1
    ]
    return " ".join(tokens).strip()

# --- 4. Order Matching Logic (for single item) ---
def match_single_item(customer_phrase_input, customer_id, df_orders, nlp_model):
    customer_data = df_orders[df_orders['customer_id'] == customer_id].copy()

    if customer_data.empty:
        return "Error: Customer ID not found or no specific orders for this customer in the database.", None, None

    customer_name = customer_data['customer_name'].iloc[0]
    normalized_message = normalize_text(customer_phrase_input, nlp_model)

    # Handle cases where input becomes empty after normalization (e.g., "1 case" -> "")
    if not normalized_message:
        return (f"**No Meaningful Content:** '{customer_phrase_input}' was normalized to an empty string. Cannot match."), None, customer_name

    # 4a. Try Direct String Match on Normalized Phrases (Highest Confidence / "Green" equivalent)
    for index, row in customer_data.iterrows():
        normalized_legend_phrase = normalize_text(row['customer_phrase'], nlp_model)
        if normalized_message == normalized_legend_phrase:
            return (f"**EXACT Match (Normalized):** "
                    f"Input: '{customer_phrase_input}' (cleaned: '{normalized_message}') matched Legend: '{row['customer_phrase']}' (cleaned: '{normalized_legend_phrase}')"), row['internal_sku'], customer_name

    # 4b. Fallback to Semantic Similarity
    customer_data['normalized_phrase'] = customer_data['customer_phrase'].apply(lambda x: normalize_text(x, nlp_model))
    valid_customer_data = customer_data[customer_data['normalized_phrase'].str.len() > 0]

    if valid_customer_data.empty:
        return (f"No valid phrases for Customer ID '{customer_id}' after normalization. Cannot perform semantic matching."), None, customer_name

    vectorizer = TfidfVectorizer()
    all_texts_for_vectorization = valid_customer_data['normalized_phrase'].tolist() + [normalized_message]

    if len(all_texts_for_vectorization) < 2 or not any(len(t) > 0 for t in all_texts_for_vectorization):
        return (f"Not enough meaningful data to perform semantic matching for Customer ID '{customer_id}'. Please add more descriptive phrases."), None, customer_name

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts_for_vectorization)
    except ValueError as e:
        return (f"Could not vectorize phrases for semantic matching. Ensure customer phrases are meaningful. Error: {e}"), None, customer_name

    message_vector = tfidf_matrix[-1:]
    phrase_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(message_vector, phrase_vectors).flatten()

    best_match_idx = similarities.argmax()
    max_similarity = similarities[best_match_idx]
    best_matched_original_phrase = valid_customer_data.iloc[best_match_idx]['customer_phrase']
    predicted_sku = valid_customer_data.iloc[best_match_idx]['internal_sku']
    best_matched_normalized_phrase = valid_customer_data.iloc[best_match_idx]['normalized_phrase']

    EXACT_SEMANTIC_THRESHOLD = 0.95
    POTENTIAL_MATCH_THRESHOLD = 0.65

    if max_similarity >= EXACT_SEMANTIC_THRESHOLD:
        return (f"**HIGH Confidence Semantic Match:** "
                f"Input: '{customer_phrase_input}' similar to Legend: '{best_matched_original_phrase}' "
                f"(Cleaned: '{best_matched_normalized_phrase}'). Confidence: {max_similarity:.2f}"), predicted_sku, customer_name
    elif max_similarity > POTENTIAL_MATCH_THRESHOLD:
        return (f"**Potential Semantic Match:** "
                f"Input: '{customer_phrase_input}' similar to Legend: '{best_matched_original_phrase}' "
                f"(Cleaned: '{best_matched_normalized_phrase}'). Confidence: {max_similarity:.2f}"), predicted_sku, customer_name
    else:
        return (f"**No Clear Match:** "
                f"Input: '{customer_phrase_input}'. "
                f"Closest was '{best_matched_original_phrase}' (Cleaned: '{best_matched_normalized_phrase}') with low Confidence: {max_similarity:.2f}."), None, customer_name

# --- Function to Process Full Order ---
def process_full_order(full_customer_message, customer_id, df_orders, nlp_model):
    items = [item.strip() for item in full_customer_message.split('\n') if item.strip()]

    if not items:
        st.write("No items found in the message to process.")
        return

    customer_name = df_orders[df_orders['customer_id'] == customer_id]['customer_name'].iloc[0] if not df_orders[df_orders['customer_id'] == customer_id].empty else customer_id

    st.subheader(f"Processing Order for {customer_name}:")
    st.markdown("---")

    summary_lines = []
    detail_lines = []

    for i, item_message in enumerate(items):
        result_msg, sku, _ = match_single_item(item_message, customer_id, df_orders, nlp_model)

        # 1. Prepare the concise summary line for this item
        summary_prefix = f"**Item {i+1}:** _{item_message}_ → "
        if sku:
            summary_sku_part = f"<span style='color:green; font-weight:bold;'>SKU: `{sku}`</span>"
        else:
            summary_sku_part = f"<span style='color:red; font-weight:bold;'>NO SKU FOUND</span>"
        
        summary_lines.append(summary_prefix + summary_sku_part)
        detail_lines.append(result_msg)

    # --- Display Summary Section ---
    st.markdown("#### Order Summary:")
    for line in summary_lines:
        st.markdown(line, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- Display Detailed Explanations Section ---
    st.markdown("#### Match Details:")
    for i, detail in enumerate(detail_lines):
        st.markdown(f"**{detail}**")
        if i < len(detail_lines) - 1:
            st.markdown("")

    st.markdown("---")
    st.write("Order processing complete. Please review the results above.")

# --- Helper function to get customer stats ---
def get_customer_stats(df_orders):
    """Get statistics about customers in the database"""
    stats = df_orders.groupby(['customer_id', 'customer_name']).agg({
        'customer_phrase': 'count',
        'internal_sku': 'nunique'
    }).reset_index()
    stats.columns = ['customer_id', 'customer_name', 'total_phrases', 'unique_skus']
    return stats

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Food Wholesale Order Matcher", layout="wide")
    
    # Load data
    df_orders = load_order_data()
    nlp = load_spacy_model()
    
    # App header
    st.title("🍽️ Food Wholesale Order Matching System")
    st.markdown("Select a customer and paste their order message to find matching SKUs.")
    
    # Sidebar with customer information
    with st.sidebar:
        st.header("Customer Database")
        customer_stats = get_customer_stats(df_orders)
        
        st.metric("Total Customers", len(customer_stats))
        st.metric("Total Products", df_orders['internal_sku'].nunique())
        st.metric("Total Phrases", len(df_orders))
        
        st.markdown("---")
        st.markdown("**Customer Details:**")
        for _, row in customer_stats.iterrows():
            st.markdown(f"**{row['customer_name']}** ({row['customer_id']})")
            st.markdown(f"  - {row['total_phrases']} phrases")
            st.markdown(f"  - {row['unique_skus']} unique SKUs")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Customer Selection")
        
        # Create customer options for dropdown
        customer_options = df_orders[['customer_id', 'customer_name']].drop_duplicates().sort_values('customer_name')
        customer_display = [f"{row['customer_name']} ({row['customer_id']})" for _, row in customer_options.iterrows()]
        customer_mapping = dict(zip(customer_display, customer_options['customer_id']))
        
        selected_customer_display = st.selectbox(
            "Select Customer:",
            options=customer_display,
            help="Choose the customer whose order you want to process"
        )
        
        selected_customer_id = customer_mapping[selected_customer_display]
        selected_customer_name = selected_customer_display.split(" (")[0]
        
        # Show selected customer info
        customer_data = df_orders[df_orders['customer_id'] == selected_customer_id]
        st.info(f"**Selected:** {selected_customer_name}\n\n"
                f"**Available phrases:** {len(customer_data)}\n\n"
                f"**Unique SKUs:** {customer_data['internal_sku'].nunique()}")
    
    with col2:
        st.subheader("Order Processing")
        
        with st.form("order_form"):
            customer_message_input = st.text_area(
                f"**Paste {selected_customer_name}'s Order Message:**",
                height=300,
                placeholder="Paste the customer's text message here...\n\nEach line should be a separate item.\n\nExample:\n1 beef tripe cut ready\n360 pounds noodle\n1 m.s.g\n1 peas and carrots"
            ).strip()
            
            col_a, col_b = st.columns(2)
            with col_a:
                submitted = st.form_submit_button("🔍 Process Order", type="primary")
            with col_b:
                clear_button = st.form_submit_button("🗑️ Clear")
        
        if submitted:
            if not customer_message_input:
                st.error("Please paste the customer's order message.")
            else:
                st.success(f"Processing order for {selected_customer_name}...")
                process_full_order(customer_message_input, selected_customer_id, df_orders, nlp)
        
        if clear_button:
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("- Each line in the order message is processed as a separate item")
    st.markdown("- The system uses both exact matching and semantic similarity")
    st.markdown("- Update your CSV file and restart the app to add new customers or phrases")

if __name__ == "__main__":
    main()